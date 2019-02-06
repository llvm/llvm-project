#
#//===----------------------------------------------------------------------===//
#//
#// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#// See https://llvm.org/LICENSE.txt for license information.
#// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#//
#//===----------------------------------------------------------------------===//
#
package Build;

use strict;
use warnings;

use Cwd qw{};

use LibOMP;
use tools;
use Uname;
use Platform ":vars";

my $host = Uname::host_name();
my $root = $ENV{ LIBOMP_WORK    };
my $tmp  = $ENV{ LIBOMP_TMP     };
my $out  = $ENV{ LIBOMP_EXPORTS };

my @jobs;
our $start = time();

# --------------------------------------------------------------------------------------------------
# Helper functions.
# --------------------------------------------------------------------------------------------------

# tstr -- Time string. Returns string "yyyy-dd-mm hh:mm:ss UTC".
sub tstr(;$) {
    my ( $time ) = @_;
    if ( not defined( $time ) ) {
        $time = time();
    }; # if
    my ( $sec, $min, $hour, $day, $month, $year ) = gmtime( $time );
    $month += 1;
    $year  += 1900;
    my $str = sprintf( "%04d-%02d-%02d %02d:%02d:%02d UTC", $year, $month, $day, $hour, $min, $sec );
    return $str;
}; # sub tstr

# dstr -- Duration string. Returns string "hh:mm:ss".
sub dstr($) {
    # Get time in seconds and format it as time in hours, minutes, seconds.
    my ( $sec ) = @_;
    my ( $h, $m, $s );
    $h   = int( $sec / 3600 );
    $sec = $sec - $h * 3600;
    $m   = int( $sec / 60 );
    $sec = $sec - $m * 60;
    $s   = int( $sec );
    $sec = $sec - $s;
    return sprintf( "%02d:%02d:%02d", $h, $m, $s );
}; # sub dstr

# rstr -- Result string.
sub rstr($) {
    my ( $rc ) = @_;
    return ( $rc == 0 ? "+++ Success +++" : "--- Failure ---" );
}; # sub rstr

sub shorter($;$) {
    # Return shorter variant of path -- either absolute or relative.
    my ( $path, $base ) = @_;
    my $abs = abs_path( $path );
    my $rel = rel_path( $path, $base );
    if ( $rel eq "" ) {
        $rel = ".";
    }; # if
    $path = ( length( $rel ) < length( $abs ) ? $rel : $abs );
    if ( $target_os eq "win" ) {
        $path =~ s{\\}{/}g;
    }; # if
    return $path;
}; # sub shorter

sub tee($$) {

    my ( $action, $file ) = @_;
    my $pid = 0;

    my $save_stdout = Symbol::gensym();
    my $save_stderr = Symbol::gensym();

    # --- redirect stdout ---
    STDOUT->flush();
    # Save stdout in $save_stdout.
    open( $save_stdout, ">&" . STDOUT->fileno() )
        or die( "Cannot dup filehandle: $!; stopped" );
    # Redirect stdout to tee or to file.
    if ( $tools::verbose ) {
        $pid = open( STDOUT, "| tee -a \"$file\"" )
            or die "Cannot open pipe to \"tee\": $!; stopped";
    } else {
        open( STDOUT, ">>$file" )
            or die "Cannot open file \"$file\" for writing: $!; stopped";
    }; # if

    # --- redirect stderr ---
    STDERR->flush();
    # Save stderr in $save_stderr.
    open( $save_stderr, ">&" . STDERR->fileno() )
        or die( "Cannot dup filehandle: $!; stopped" );
    # Redirect stderr to stdout.
    open( STDERR, ">&" . STDOUT->fileno() )
        or die( "Cannot dup filehandle: $!; stopped" );

    # Perform actions.
    $action->();

    # --- restore stderr ---
    STDERR->flush();
    # Restore stderr from $save_stderr.
    open( STDERR, ">&" . $save_stderr->fileno() )
        or die( "Cannot dup filehandle: $!; stopped" );
    # Close $save_stderr.
    $save_stderr->close() or die ( "Cannot close filehandle: $!; stopped" );

    # --- restore stdout ---
    STDOUT->flush();
    # Restore stdout from $save_stdout.
    open( STDOUT, ">&" . $save_stdout->fileno() )
        or die( "Cannot dup filehandle: $!; stopped" );
    # Close $save_stdout.
    $save_stdout->close() or die ( "Cannot close filehandle: $!; stopped" );

    # Wait for the child tee process, otherwise output of make and build.pl interleaves.
    if ( $pid != 0 ) {
        waitpid( $pid, 0 );
    }; # if

}; # sub tee

sub log_it($$@) {
    my ( $title, $format, @args ) = @_;
    my $message  = sprintf( $format, @args );
    my $progress = cat_file( $tmp, sprintf( "%s-%s.log", $target_platform, Uname::host_name() ) );
    if ( $title ne "" and $message ne "" ) {
        my $line = sprintf( "%-15s : %s\n", $title, $message );
        info( $line );
        write_file( $progress, tstr() . ": " . $line, -append => 1 );
    } else {
        write_file( $progress, "\n", -append => 1 );
    }; # if
}; # sub log_it

sub progress($$@) {
    my ( $title, $format, @args ) = @_;
    log_it( $title, $format, @args );
}; # sub progress

sub summary() {
    my $total   = @jobs;
    my $success = 0;
    my $finish = time();
    foreach my $job ( @jobs ) {
        my ( $build_dir, $rc ) = ( $job->{ build_dir }, $job->{ rc } );
        progress( rstr( $rc ), "%s", $build_dir );
        if ( $rc == 0 ) {
            ++ $success;
        }; # if
    }; # foreach $job
    my $failure = $total - $success;
    progress( "Successes",      "%3d of %3d", $success, $total );
    progress( "Failures",       "%3d of %3d", $failure, $total );
    progress( "Time elapsed",   "  %s", dstr( $finish - $start ) );
    progress( "Overall result", "%s", rstr( $failure ) );
    return $failure;
}; # sub summary

# --------------------------------------------------------------------------------------------------
# Worker functions.
# --------------------------------------------------------------------------------------------------

sub init() {
    make_dir( $tmp );
}; # sub init

sub clean(@) {
    # Clean directories.
    my ( @dirs ) = @_;
    my $exit = 0;
    # Mimisc makefile -- print a command.
    print( "rm -f -r " . join( " ", map( shorter( $_ ) . "/*", @dirs ) ) . "\n" );
    $exit =
        execute(
            [ $^X, cat_file( $ENV{ LIBOMP_WORK }, "tools", "clean-dir.pl" ), @dirs ],
            -ignore_status => 1,
            ( $tools::verbose ? () : ( -stdout => undef, -stderr => "" ) ),
        );
    return $exit;
}; # sub clean

sub make($$$) {
    # Change dir to build one and run make.
    my ( $job, $clean, $marker ) = @_;
    my $dir      = $job->{ build_dir };
    my $makefile = $job->{ makefile };
    my $args     = $job->{ make_args };
    my $cwd      = Cwd::cwd();
    my $width    = -10;

    my $exit;
    $dir = cat_dir( $tmp, $dir );
    make_dir( $dir );
    change_dir( $dir );

    my $actions =
        sub {
            my $start = time();
            $makefile = shorter( $makefile );
            print( "-" x 79, "\n" );
            printf( "%${width}s: %s\n", "Started",   tstr( $start ) );
            printf( "%${width}s: %s\n", "Root dir",  $root );
            printf( "%${width}s: %s\n", "Build dir", shorter( $dir, $root ) );
            printf( "%${width}s: %s\n", "Makefile",  $makefile );
            print( "-" x 79, "\n" );
            {
                # Use shorter LIBOMP_WORK to have shorter command lines.
                # Note: Some tools may not work if current dir is changed.
                local $ENV{ LIBOMP_WORK } = shorter( $ENV{ LIBOMP_WORK } );
                $exit =
                    execute(
                        [
                            "make",
                            "-r",
                            "-f", $makefile,
                            "arch=" . $target_arch,
                            "marker=$marker",
                            @$args
                        ],
                        -ignore_status => 1
                    );
                if ( $clean and $exit == 0 ) {
                    $exit = clean( $dir );
                }; # if
            }
            my $finish = time();
            print( "-" x 79, "\n" );
            printf( "%${width}s: %s\n", "Finished", tstr( $finish ) );
            printf( "%${width}s: %s\n", "Elapsed", dstr( $finish - $start ) );
            printf( "%${width}s: %s\n", "Result", rstr( $exit ) );
            print( "-" x 79, "\n" );
            print( "\n" );
        }; # sub
    tee( $actions, "build.log" );

    change_dir( $cwd );

    # Save completed job to be able print summary later.
    $job->{ rc } = $exit;
    push( @jobs, $job );

    return $exit;

}; # sub make

1;
