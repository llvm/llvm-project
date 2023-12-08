#
# This is not a runnable script, it is a Perl module, a collection of variables, subroutines, etc.
# to be used in other scripts.
#
# To get help about exported variables and subroutines, please execute the following command:
#
#     perldoc tools.pm
#
# or see POD (Plain Old Documentation) imbedded to the source...
#
#
#//===----------------------------------------------------------------------===//
#//
#// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#// See https://llvm.org/LICENSE.txt for license information.
#// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#//
#//===----------------------------------------------------------------------===//
#

=head1 NAME

B<tools.pm> -- A collection of subroutines which are widely used in Perl scripts.

=head1 SYNOPSIS

    use FindBin;
    use lib "$FindBin::Bin/lib";
    use tools;

=head1 DESCRIPTION

B<Note:> Because this collection is small and intended for widely using in particular project,
all variables and functions are exported by default.

B<Note:> I have some ideas how to improve this collection, but it is in my long-term plans.
Current shape is not ideal, but good enough to use.

=cut

package tools;

use strict;
use warnings;

use vars qw( @ISA @EXPORT @EXPORT_OK %EXPORT_TAGS );
require Exporter;
@ISA = qw( Exporter );

my @vars   = qw( $tool );
my @utils  = qw( check_opts validate );
my @opts   = qw( get_options );
my @print  = qw( debug info warning cmdline_error runtime_error question );
my @name   = qw( get_vol get_dir get_file get_name get_ext cat_file cat_dir );
my @file   = qw( which abs_path rel_path real_path make_dir clean_dir copy_dir move_dir del_dir change_dir copy_file move_file del_file );
my @io     = qw( read_file write_file );
my @exec   = qw( execute backticks );
my @string = qw{ pad };
@EXPORT = ( @utils, @opts, @vars, @print, @name, @file, @io, @exec, @string );

use UNIVERSAL    ();

use FindBin;
use IO::Handle;
use IO::File;
use IO::Dir;
# Not available on some machines: use IO::Zlib;

use Getopt::Long ();
use Pod::Usage   ();
use Carp         ();
use File::Copy   ();
use File::Path   ();
use File::Temp   ();
use File::Spec   ();
use POSIX        qw{ :fcntl_h :errno_h };
use Cwd          ();
use Symbol       ();

use Data::Dumper;

use vars qw( $tool $verbose $timestamps );
$tool = $FindBin::Script;

my @warning = ( sub {}, \&warning, \&runtime_error );


sub check_opts(\%$;$) {

    my $opts = shift( @_ );  # Reference to hash containing real options and their values.
    my $good = shift( @_ );  # Reference to an array containing all known option names.
    my $msg  = shift( @_ );  # Optional (non-mandatory) message.

    if ( not defined( $msg ) ) {
        $msg = "unknown option(s) passed";   # Default value for $msg.
    }; # if

    # I'll use these hashes as sets of options.
    my %good = map( ( $_ => 1 ), @$good );   # %good now is filled with all known options.
    my %bad;                                 # %bad is empty.

    foreach my $opt ( keys( %$opts ) ) {     # For each real option...
        if ( not exists( $good{ $opt } ) ) { # Look its name in the set of known options...
            $bad{ $opt } = 1;                # Add unknown option to %bad set.
            delete( $opts->{ $opt } );       # And delete original option.
        }; # if
    }; # foreach $opt
    if ( %bad ) {                            # If %bad set is not empty...
        my @caller = caller( 1 );            # Issue a warning.
        local $Carp::CarpLevel = 2;
        Carp::cluck( $caller[ 3 ] . ": " . $msg . ": " . join( ", ", sort( keys( %bad ) ) ) );
    }; # if

    return 1;

}; # sub check_opts


# --------------------------------------------------------------------------------------------------
# Purpose:
#     Check subroutine arguments.
# Synopsis:
#     my %opts = validate( params => \@_, spec => { ... }, caller => n );
# Arguments:
#     params -- A reference to subroutine's actual arguments.
#     spec   -- Specification of expected arguments.
#     caller -- ...
# Return value:
#     A hash of validated options.
# Description:
#     I would like to use Params::Validate module, but it is not a part of default Perl
#     distribution, so I cannot rely on it. This subroutine resembles to some extent to
#     Params::Validate::validate_with().
#     Specification of expected arguments:
#        { $opt => { type => $type, default => $default }, ... }
#        $opt     -- String, option name.
#        $type    -- String, expected type(s). Allowed values are "SCALAR", "UNDEF", "BOOLEAN",
#                    "ARRAYREF", "HASHREF", "CODEREF". Multiple types may listed using bar:
#                    "SCALAR|ARRAYREF". The type string is case-insensitive.
#        $default -- Default value for an option. Will be used if option is not specified or
#                    undefined.
#
sub validate(@) {

    my %opts = @_;    # Temporary use %opts for parameters of `validate' subroutine.
    my $params = $opts{ params };
    my $caller = ( $opts{ caller } or 0 ) + 1;
    my $spec   = $opts{ spec };
    undef( %opts );   # Ok, Clean %opts, now we will collect result of the subroutine.

    # Find out caller package, filename, line, and subroutine name.
    my ( $pkg, $file, $line, $subr ) = caller( $caller );
    my @errors;    # We will collect errors in array not to stop on the first found error.
    my $error =
        sub ($) {
            my $msg = shift( @_ );
            push( @errors, "$msg at $file line $line.\n" );
        }; # sub

    # Check options.
    while ( @$params ) {
        # Check option name.
        my $opt = shift( @$params );
        if ( not exists( $spec->{ $opt } ) ) {
            $error->( "Invalid option `$opt'" );
            shift( @$params ); # Skip value of unknow option.
            next;
        }; # if
        # Check option value exists.
        if ( not @$params ) {
            $error->( "Option `$opt' does not have a value" );
            next;
        }; # if
        my $val = shift( @$params );
        # Check option value type.
        if ( exists( $spec->{ $opt }->{ type } ) ) {
            # Type specification exists. Check option value type.
            my $actual_type;
            if ( ref( $val ) ne "" ) {
                $actual_type = ref( $val ) . "REF";
            } else {
                $actual_type = ( defined( $val ) ? "SCALAR" : "UNDEF" );
            }; # if
            my @wanted_types = split( m{\|}, lc( $spec->{ $opt }->{ type } ) );
            my $wanted_types = join( "|", map( $_ eq "boolean" ? "scalar|undef" : quotemeta( $_ ), @wanted_types ) );
            if ( $actual_type !~ m{\A(?:$wanted_types)\z}i ) {
                $actual_type = lc( $actual_type );
                $wanted_types = lc( join( " or ", map( "`$_'", @wanted_types ) ) );
                $error->( "Option `$opt' value type is `$actual_type' but expected to be $wanted_types" );
                next;
            }; # if
        }; # if
        if ( exists( $spec->{ $opt }->{ values } )  ) {
            my $values = $spec->{ $opt }->{ values };
            if ( not grep( $_ eq $val, @$values ) ) {
                $values = join( ", ", map( "`$_'", @$values ) );
                $error->( "Option `$opt' value is `$val' but expected to be one of $values" );
                next;
            }; # if
        }; # if
        $opts{ $opt } = $val;
    }; # while

    # Assign default values.
    foreach my $opt ( keys( %$spec ) ) {
        if ( not defined( $opts{ $opt } ) and exists( $spec->{ $opt }->{ default } ) ) {
            $opts{ $opt } = $spec->{ $opt }->{ default };
        }; # if
    }; # foreach $opt

    # If we found any errors, raise them.
    if ( @errors ) {
        die join( "", @errors );
    }; # if

    return %opts;

}; # sub validate

# =================================================================================================
# Get option helpers.
# =================================================================================================

=head2 Get option helpers.

=cut

# -------------------------------------------------------------------------------------------------

=head3 get_options

B<Synopsis:>

    get_options( @arguments )

B<Description:>

It is very simple wrapper arounf Getopt::Long::GetOptions. It passes all arguments to GetOptions,
and add definitions for standard help options: --help, --doc, --verbose, and --quiet.
When GetOptions finishes, this subroutine checks exit code, if it is non-zero, standard error
message is issued and script terminated.

If --verbose or --quiet option is specified, C<tools.pm_verbose> environment variable is set.
It is the way to propagate verbose/quiet mode to callee Perl scripts.

=cut

sub get_options {

    Getopt::Long::Configure( "no_ignore_case" );
    Getopt::Long::GetOptions(
        "h0|usage"        => sub { Pod::Usage::pod2usage( -exitval => 0, -verbose => 0 ); },
        "h1|h|help"       => sub { Pod::Usage::pod2usage( -exitval => 0, -verbose => 1 ); },
        "h2|doc|manual"   => sub { Pod::Usage::pod2usage( -exitval => 0, -verbose => 2 ); },
        "version"         => sub { print( "$tool version $main::VERSION\n" ); exit( 0 ); },
        "v|verbose"       => sub { ++ $verbose;     $ENV{ "tools.pm_verbose"    } = $verbose;    },
        "quiet"           => sub { -- $verbose;     $ENV{ "tools.pm_verbose"    } = $verbose;    },
        "with-timestamps" => sub { $timestamps = 1; $ENV{ "tools.pm_timestamps" } = $timestamps; },
        @_, # Caller arguments are at the end so caller options overrides standard.
    ) or cmdline_error();

}; # sub get_options


# =================================================================================================
# Print utilities.
# =================================================================================================

=pod

=head2 Print utilities.

Each of the print subroutines prepends each line of its output with the name of current script and
the type of information, for example:

    info( "Writing file..." );

will print

    <script>: (i): Writing file...

while

    warning( "File does not exist!" );

will print

    <script>: (!): File does not exist!

Here are exported items:

=cut

# -------------------------------------------------------------------------------------------------

sub _format_message($\@;$) {

    my $prefix  = shift( @_ );
    my $args    = shift( @_ );
    my $no_eol  = shift( @_ );  # Do not append "\n" to the last line.
    my $message = "";

    my $ts = "";
    if ( $timestamps ) {
        my ( $sec, $min, $hour, $day, $month, $year ) = gmtime();
        $month += 1;
        $year  += 1900;
        $ts = sprintf( "%04d-%02d-%02d %02d:%02d:%02d UTC: ", $year, $month, $day, $hour, $min, $sec );
    }; # if
    for my $i ( 1 .. @$args ) {
        my @lines = split( "\n", $args->[ $i - 1 ] );
        for my $j ( 1 .. @lines ) {
            my $line = $lines[ $j - 1 ];
            my $last_line = ( ( $i == @$args ) and ( $j == @lines ) );
            my $eol = ( ( substr( $line, -1 ) eq "\n" ) or defined( $no_eol ) ? "" : "\n" );
            $message .= "$ts$tool: ($prefix) " . $line . $eol;
        }; # foreach $j
    }; # foreach $i
    return $message;

}; # sub _format_message

#--------------------------------------------------------------------------------------------------

=pod

=head3 $verbose

B<Synopsis:>

    $verbose

B<Description:>

Package variable. It determines verbosity level, which affects C<warning()>, C<info()>, and
C<debug()> subroutines .

The variable gets initial value from C<tools.pm_verbose> environment variable if it is exists.
If the environment variable does not exist, variable is set to 2.

Initial value may be overridden later directly or by C<get_options> function.

=cut

$verbose = exists( $ENV{ "tools.pm_verbose" } ) ? $ENV{ "tools.pm_verbose" } : 2;

#--------------------------------------------------------------------------------------------------

=pod

=head3 $timestamps

B<Synopsis:>

    $timestamps

B<Description:>

Package variable. It determines whether C<debug()>, C<info()>, C<warning()>, C<runtime_error()>
subroutines print timestamps or not.

The variable gets initial value from C<tools.pm_timestamps> environment variable if it is exists.
If the environment variable does not exist, variable is set to false.

Initial value may be overridden later directly or by C<get_options()> function.

=cut

$timestamps = exists( $ENV{ "tools.pm_timestamps" } ) ? $ENV{ "tools.pm_timestamps" } : 0;

# -------------------------------------------------------------------------------------------------

=pod

=head3 debug

B<Synopsis:>

    debug( @messages )

B<Description:>

If verbosity level is 3 or higher, print debug information to the stderr, prepending it with "(#)"
prefix.

=cut

sub debug(@) {

    if ( $verbose >= 3 ) {
        STDOUT->flush();
        STDERR->print( _format_message( "#", @_ ) );
    }; # if
    return 1;

}; # sub debug

#--------------------------------------------------------------------------------------------------

=pod

=head3 info

B<Synopsis:>

    info( @messages )

B<Description:>

If verbosity level is 2 or higher, print information to the stderr, prepending it with "(i)" prefix.

=cut

sub info(@) {

    if ( $verbose >= 2 ) {
        STDOUT->flush();
        STDERR->print( _format_message( "i", @_  ) );
    }; # if

}; # sub info

#--------------------------------------------------------------------------------------------------

=head3 warning

B<Synopsis:>

    warning( @messages )

B<Description:>

If verbosity level is 1 or higher, issue a warning, prepending it with "(!)" prefix.

=cut

sub warning(@) {

    if ( $verbose >= 1 ) {
        STDOUT->flush();
        warn( _format_message( "!", @_  ) );
    }; # if

}; # sub warning

# -------------------------------------------------------------------------------------------------

=head3 cmdline_error

B<Synopsis:>

    cmdline_error( @message )

B<Description:>

Print error message and exit the program with status 2.

This function is intended to complain on command line errors, e. g. unknown
options, invalid arguments, etc.

=cut

sub cmdline_error(;$) {

    my $message = shift( @_ );

    if ( defined( $message ) ) {
        if ( substr( $message, -1, 1 ) ne "\n" ) {
            $message .= "\n";
        }; # if
    } else {
        $message = "";
    }; # if
    STDOUT->flush();
    die $message . "Try --help option for more information.\n";

}; # sub cmdline_error

# -------------------------------------------------------------------------------------------------

=head3 runtime_error

B<Synopsis:>

    runtime_error( @message )

B<Description:>

Print error message and exits the program with status 3.

This function is intended to complain on runtime errors, e. g.
directories which are not found, non-writable files, etc.

=cut

sub runtime_error(@) {

    STDOUT->flush();
    die _format_message( "x", @_ );

}; # sub runtime_error

#--------------------------------------------------------------------------------------------------

=head3 question

B<Synopsis:>

    question( $prompt; $answer, $choices  )

B<Description:>

Print $promp to the stderr, prepending it with "question:" prefix. Read a line from stdin, chop
"\n" from the end, it is answer.

If $answer is defined, it is treated as first user input.

If $choices is specified, it could be a regexp for validating user input, or a string. In latter
case it interpreted as list of characters, acceptable (case-insensitive) choices. If user enters
non-acceptable answer, question continue asking until answer is acceptable.
If $choices is not specified, any answer is acceptable.

In case of end-of-file (or Ctrl+D pressed by user), $answer is C<undef>.

B<Examples:>

    my $answer;
    question( "Save file [yn]? ", $answer, "yn" );
        # We accepts only "y", "Y", "n", or "N".
    question( "Press enter to continue or Ctrl+C to abort..." );
        # We are not interested in answer value -- in case of Ctrl+C the script will be terminated,
        # otherwise we continue execution.
    question( "File name? ", $answer );
        # Any answer is acceptable.

=cut

sub question($;\$$) {

    my $prompt  = shift( @_ );
    my $answer  = shift( @_ );
    my $choices = shift( @_ );
    my $a       = ( defined( $answer ) ? $$answer : undef );

    if ( ref( $choices ) eq "Regexp" ) {
        # It is already a regular expression, do nothing.
    } elsif ( defined( $choices ) ) {
        # Convert string to a regular expression.
        $choices = qr/[@{ [ quotemeta( $choices ) ] }]/i;
    }; # if

    for ( ; ; ) {
        STDERR->print( _format_message( "?", @{ [ $prompt ] }, "no_eol" ) );
        STDERR->flush();
        if ( defined( $a ) ) {
            STDOUT->print( $a . "\n" );
        } else {
            $a = <STDIN>;
        }; # if
        if ( not defined( $a ) ) {
            last;
        }; # if
        chomp( $a );
        if ( not defined( $choices ) or ( $a =~ m/^$choices$/ ) ) {
            last;
        }; # if
        $a = undef;
    }; # forever
    if ( defined( $answer ) ) {
        $$answer = $a;
    }; # if

}; # sub question

# -------------------------------------------------------------------------------------------------

# Returns volume part of path.
sub get_vol($) {

    my $path = shift( @_ );
    my ( $vol, undef, undef ) = File::Spec->splitpath( $path );
    return $vol;

}; # sub get_vol

# Returns directory part of path.
sub get_dir($) {

    my $path = File::Spec->canonpath( shift( @_ ) );
    my ( $vol, $dir, undef ) = File::Spec->splitpath( $path );
    my @dirs = File::Spec->splitdir( $dir );
    pop( @dirs );
    $dir = File::Spec->catdir( @dirs );
    $dir = File::Spec->catpath( $vol, $dir, undef );
    return $dir;

}; # sub get_dir

# Returns file part of path.
sub get_file($) {

    my $path = shift( @_ );
    my ( undef, undef, $file ) = File::Spec->splitpath( $path );
    return $file;

}; # sub get_file

# Returns file part of path without last suffix.
sub get_name($) {

    my $path = shift( @_ );
    my ( undef, undef, $file ) = File::Spec->splitpath( $path );
    $file =~ s{\.[^.]*\z}{};
    return $file;

}; # sub get_name

# Returns last suffix of file part of path.
sub get_ext($) {

    my $path = shift( @_ );
    my ( undef, undef, $file ) = File::Spec->splitpath( $path );
    my $ext = "";
    if ( $file =~ m{(\.[^.]*)\z} ) {
        $ext = $1;
    }; # if
    return $ext;

}; # sub get_ext

sub cat_file(@) {

    my $path = shift( @_ );
    my $file = pop( @_ );
    my @dirs = @_;

    my ( $vol, $dirs ) = File::Spec->splitpath( $path, "no_file" );
    @dirs = ( File::Spec->splitdir( $dirs ), @dirs );
    $dirs = File::Spec->catdir( @dirs );
    $path = File::Spec->catpath( $vol, $dirs, $file );

    return $path;

}; # sub cat_file

sub cat_dir(@) {

    my $path = shift( @_ );
    my @dirs = @_;

    my ( $vol, $dirs ) = File::Spec->splitpath( $path, "no_file" );
    @dirs = ( File::Spec->splitdir( $dirs ), @dirs );
    $dirs = File::Spec->catdir( @dirs );
    $path = File::Spec->catpath( $vol, $dirs, "" );

    return $path;

}; # sub cat_dir

# =================================================================================================
# File and directory manipulation subroutines.
# =================================================================================================

=head2 File and directory manipulation subroutines.

=over

=cut

# -------------------------------------------------------------------------------------------------

=item C<which( $file, @options )>

Searches for specified executable file in the (specified) directories.
Raises a runtime eroror if no executable file found. Returns a full path of found executable(s).

Options:

=over

=item C<-all> =E<gt> I<bool>

Do not stop on the first found file. Note, that list of full paths is returned in this case.

=item C<-dirs> =E<gt> I<ref_to_array>

Specify directory list to search through. If option is not passed, PATH environment variable
is used for directory list.

=item C<-exec> =E<gt> I<bool>

Whether check for executable files or not. By default, C<which> searches executable files.
However, on Cygwin executable check never performed.

=back

Examples:

Look for "echo" in the directories specified in PATH:

    my $echo = which( "echo" );

Look for all occurrences of "cp" in the PATH:

    my @cps = which( "cp", -all => 1 );

Look for the first occurrence of "icc" in the specified directories:

    my $icc = which( "icc", -dirs => [ ".", "/usr/local/bin", "/usr/bin", "/bin" ] );

=cut

sub which($@) {

    my $file = shift( @_ );
    my %opts = @_;

    check_opts( %opts, [ qw( -all -dirs -exec ) ] );
    if ( $opts{ -all } and not wantarray() ) {
        local $Carp::CarpLevel = 1;
        Carp::cluck( "`-all' option passed to `which' but list is not expected" );
    }; # if
    if ( not defined( $opts{ -exec } ) ) {
        $opts{ -exec } = 1;
    }; # if

    my $dirs = ( exists( $opts{ -dirs } ) ? $opts{ -dirs } : [ File::Spec->path() ] );
    my @found;

    my @exts = ( "" );
    if ( $^O eq "MSWin32" and $opts{ -exec } ) {
        if ( defined( $ENV{ PATHEXT } ) ) {
            push( @exts, split( ";", $ENV{ PATHEXT } ) );
        } else {
            # If PATHEXT does not exist, use default value.
            push( @exts, qw{ .COM .EXE .BAT .CMD } );
        }; # if
    }; # if

    loop:
    foreach my $dir ( @$dirs ) {
        foreach my $ext ( @exts ) {
            my $path = File::Spec->catfile( $dir, $file . $ext );
            if ( -e $path ) {
                # Executable bit is not reliable on Cygwin, do not check it.
                if ( not $opts{ -exec } or -x $path or $^O eq "cygwin" ) {
                    push( @found, $path );
                    if ( not $opts{ -all } ) {
                        last loop;
                    }; # if
                }; # if
            }; # if
        }; # foreach $ext
    }; # foreach $dir

    if ( not @found ) {
        # TBD: We need to introduce an option for conditional enabling this error.
        # runtime_error( "Could not find \"$file\" executable file in PATH." );
    }; # if
    if ( @found > 1 ) {
        # TBD: Issue a warning?
    }; # if

    if ( $opts{ -all } ) {
        return @found;
    } else {
        return $found[ 0 ];
    }; # if

}; # sub which

# -------------------------------------------------------------------------------------------------

=item C<abs_path( $path, $base )>

Return absolute path for an argument.

Most of the work is done by C<File::Spec->rel2abs()>. C<abs_path()> additionally collapses
C<dir1/../dir2> to C<dir2>.

It is not so naive and made intentionally. For example on Linux* OS in Bash if F<link/> is a symbolic
link to directory F<some_dir/>

    $ cd link
    $ cd ..

brings you back to F<link/>'s parent, not to parent of F<some_dir/>,

=cut

sub abs_path($;$) {

    my ( $path, $base ) = @_;
    $path = File::Spec->rel2abs( $path, ( defined( $base ) ? $base : $ENV{ PWD } ) );
    my ( $vol, $dir, $file ) = File::Spec->splitpath( $path );
    while ( $dir =~ s{/(?!\.\.)[^/]*/\.\.(?:/|\z)}{/} ) {
    }; # while
    $path = File::Spec->canonpath( File::Spec->catpath( $vol, $dir, $file ) );
    return $path;

}; # sub abs_path

# -------------------------------------------------------------------------------------------------

=item C<rel_path( $path, $base )>

Return relative path for an argument.

=cut

sub rel_path($;$) {

    my ( $path, $base ) = @_;
    $path = File::Spec->abs2rel( abs_path( $path ), $base );
    return $path;

}; # sub rel_path

# -------------------------------------------------------------------------------------------------

=item C<real_path( $dir )>

Return real absolute path for an argument. In the result all relative components (F<.> and F<..>)
and U<symbolic links are resolved>.

In most cases it is not what you want. Consider using C<abs_path> first.

C<abs_path> function from B<Cwd> module works with directories only. This function works with files
as well. But, if file is a symbolic link, function does not resolve it (yet).

The function uses C<runtime_error> to raise an error if something wrong.

=cut

sub real_path($) {

    my $orig_path = shift( @_ );
    my $real_path;
    my $message = "";
    if ( not -e $orig_path ) {
        $message = "\"$orig_path\" does not exists";
    } else {
        # Cwd::abs_path does not work with files, so in this case we should handle file separately.
        my $file;
        if ( not -d $orig_path ) {
            ( my $vol, my $dir, $file ) = File::Spec->splitpath( File::Spec->rel2abs( $orig_path ) );
            $orig_path = File::Spec->catpath( $vol, $dir );
        }; # if
        {
            local $SIG{ __WARN__ } = sub { $message = $_[ 0 ]; };
            $real_path = Cwd::abs_path( $orig_path );
        };
        if ( defined( $file ) ) {
            $real_path = File::Spec->catfile( $real_path, $file );
        }; # if
    }; # if
    if ( not defined( $real_path ) or $message ne "" ) {
        $message =~ s/^stat\(.*\): (.*)\s+at .*? line \d+\s*\z/$1/;
        runtime_error( "Could not find real path for \"$orig_path\"" . ( $message ne "" ? ": $message" : "" ) );
    }; # if
    return $real_path;

}; # sub real_path

# -------------------------------------------------------------------------------------------------

=item C<make_dir( $dir, @options )>

Make a directory.

This function makes a directory. If necessary, more than one level can be created.
If directory exists, warning issues (the script behavior depends on value of
C<-warning_level> option). If directory creation fails or C<$dir> exists but it is not a
directory, error issues.

Options:

=over

=item C<-mode>

The numeric mode for new directories, 0750 (rwxr-x---) by default.

=back

=cut

sub make_dir($@) {

    my $dir    = shift( @_ );
    my %opts   =
        validate(
            params => \@_,
            spec => {
                parents => { type => "boolean", default => 1    },
                mode    => { type => "scalar",  default => 0777 },
            },
        );

    my $prefix = "Could not create directory \"$dir\"";

    if ( -e $dir ) {
        if ( -d $dir ) {
        } else {
            runtime_error( "$prefix: it exists, but not a directory." );
        }; # if
    } else {
        eval {
            File::Path::mkpath( $dir, 0, $opts{ mode } );
        }; # eval
        if ( $@ ) {
            $@ =~ s{\s+at (?:[a-zA-Z0-9 /_.]*/)?tools\.pm line \d+\s*}{};
            runtime_error( "$prefix: $@" );
        }; # if
        if ( not -d $dir ) { # Just in case, check it one more time...
            runtime_error( "$prefix." );
        }; # if
    }; # if

}; # sub make_dir

# -------------------------------------------------------------------------------------------------

=item C<copy_dir( $src_dir, $dst_dir, @options )>

Copy directory recursively.

This function copies a directory recursively.
If source directory does not exist or not a directory, error issues.

Options:

=over

=item C<-overwrite>

Overwrite destination directory, if it exists.

=back

=cut

sub copy_dir($$@) {

    my $src  = shift( @_ );
    my $dst  = shift( @_ );
    my %opts = @_;
    my $prefix = "Could not copy directory \"$src\" to \"$dst\"";

    if ( not -e $src ) {
        runtime_error( "$prefix: \"$src\" does not exist." );
    }; # if
    if ( not -d $src ) {
        runtime_error( "$prefix: \"$src\" is not a directory." );
    }; # if
    if ( -e $dst ) {
        if ( -d $dst ) {
            if ( $opts{ -overwrite } ) {
                del_dir( $dst );
            } else {
                runtime_error( "$prefix: \"$dst\" already exists." );
            }; # if
        } else {
            runtime_error( "$prefix: \"$dst\" is not a directory." );
        }; # if
    }; # if

    execute( [ "cp", "-R", $src, $dst ] );

}; # sub copy_dir

# -------------------------------------------------------------------------------------------------

=item C<move_dir( $src_dir, $dst_dir, @options )>

Move directory.

Options:

=over

=item C<-overwrite>

Overwrite destination directory, if it exists.

=back

=cut

sub move_dir($$@) {

    my $src  = shift( @_ );
    my $dst  = shift( @_ );
    my %opts = @_;
    my $prefix = "Could not copy directory \"$src\" to \"$dst\"";

    if ( not -e $src ) {
        runtime_error( "$prefix: \"$src\" does not exist." );
    }; # if
    if ( not -d $src ) {
        runtime_error( "$prefix: \"$src\" is not a directory." );
    }; # if
    if ( -e $dst ) {
        if ( -d $dst ) {
            if ( $opts{ -overwrite } ) {
                del_dir( $dst );
            } else {
                runtime_error( "$prefix: \"$dst\" already exists." );
            }; # if
        } else {
            runtime_error( "$prefix: \"$dst\" is not a directory." );
        }; # if
    }; # if

    execute( [ "mv", $src, $dst ] );

}; # sub move_dir

# -------------------------------------------------------------------------------------------------

=item C<clean_dir( $dir, @options )>

Clean a directory: delete all the entries (recursively), but leave the directory.

Options:

=over

=item C<-force> => bool

If a directory is not writable, try to change permissions first, then clean it.

=item C<-skip> => regexp

Regexp. If a directory entry mached the regexp, it is skipped, not deleted. (As a subsequence,
a directory containing skipped entries is not deleted.)

=back

=cut

sub _clean_dir($);

sub _clean_dir($) {
    our %_clean_dir_opts;
    my ( $dir ) = @_;
    my $skip    = $_clean_dir_opts{ skip };    # Regexp.
    my $skipped = 0;                           # Number of skipped files.
    my $prefix  = "Cleaning `$dir' failed:";
    my @stat    = stat( $dir );
    my $mode    = $stat[ 2 ];
    if ( not @stat ) {
        runtime_error( $prefix, "Cannot stat `$dir': $!" );
    }; # if
    if ( not -d _ ) {
        runtime_error( $prefix, "It is not a directory." );
    }; # if
    if ( not -w _ ) {        # Directory is not writable.
        if ( not -o _ or not $_clean_dir_opts{ force } ) {
            runtime_error( $prefix, "Directory is not writable." );
        }; # if
        # Directory is not writable but mine. Try to change permissions.
        chmod( $mode | S_IWUSR, $dir )
            or runtime_error( $prefix, "Cannot make directory writable: $!" );
    }; # if
    my $handle   = IO::Dir->new( $dir ) or runtime_error( $prefix, "Cannot read directory: $!" );
    my @entries  = File::Spec->no_upwards( $handle->read() );
    $handle->close() or runtime_error( $prefix, "Cannot read directory: $!" );
    foreach my $entry ( @entries ) {
        my $path = cat_file( $dir, $entry );
        if ( defined( $skip ) and $entry =~ $skip ) {
            ++ $skipped;
        } else {
            if ( -l $path ) {
                unlink( $path ) or runtime_error( $prefix, "Cannot delete symlink `$path': $!" );
            } else {
                stat( $path ) or runtime_error( $prefix, "Cannot stat `$path': $! " );
                if ( -f _ ) {
                    del_file( $path );
                } elsif ( -d _ ) {
                    my $rc = _clean_dir( $path );
                    if ( $rc == 0 ) {
                        rmdir( $path ) or runtime_error( $prefix, "Cannot delete directory `$path': $!" );
                    }; # if
                    $skipped += $rc;
                } else {
                    runtime_error( $prefix, "`$path' is neither a file nor a directory." );
                }; # if
            }; # if
        }; # if
    }; # foreach
    return $skipped;
}; # sub _clean_dir


sub clean_dir($@) {
    my $dir  = shift( @_ );
    our %_clean_dir_opts;
    local %_clean_dir_opts =
        validate(
            params => \@_,
            spec => {
                skip  => { type => "regexpref" },
                force => { type => "boolean"   },
            },
        );
    my $skipped = _clean_dir( $dir );
    return $skipped;
}; # sub clean_dir


# -------------------------------------------------------------------------------------------------

=item C<del_dir( $dir, @options )>

Delete a directory recursively.

This function deletes a directory. If directory can not be deleted or it is not a directory, error
message issues (and script exists).

Options:

=over

=back

=cut

sub del_dir($@) {

    my $dir  = shift( @_ );
    my %opts = @_;
    my $prefix = "Deleting directory \"$dir\" failed";
    our %_clean_dir_opts;
    local %_clean_dir_opts =
        validate(
            params => \@_,
            spec => {
                force => { type => "boolean" },
            },
        );

    if ( not -e $dir ) {
        # Nothing to do.
        return;
    }; # if
    if ( not -d $dir ) {
        runtime_error( "$prefix: it is not a directory." );
    }; # if
    _clean_dir( $dir );
    rmdir( $dir ) or runtime_error( "$prefix." );

}; # sub del_dir

# -------------------------------------------------------------------------------------------------

=item C<change_dir( $dir )>

Change current directory.

If any error occurred, error issues and script exits.

=cut

sub change_dir($) {

    my $dir = shift( @_ );

    Cwd::chdir( $dir )
        or runtime_error( "Could not chdir to \"$dir\": $!" );

}; # sub change_dir


# -------------------------------------------------------------------------------------------------

=item C<copy_file( $src_file, $dst_file, @options )>

Copy file.

This function copies a file. If source does not exist or is not a file, error issues.

Options:

=over

=item C<-overwrite>

Overwrite destination file, if it exists.

=back

=cut

sub copy_file($$@) {

    my $src  = shift( @_ );
    my $dst  = shift( @_ );
    my %opts = @_;
    my $prefix = "Could not copy file \"$src\" to \"$dst\"";

    if ( not -e $src ) {
        runtime_error( "$prefix: \"$src\" does not exist." );
    }; # if
    if ( not -f $src ) {
        runtime_error( "$prefix: \"$src\" is not a file." );
    }; # if
    if ( -e $dst ) {
        if ( -f $dst ) {
            if ( $opts{ -overwrite } ) {
                del_file( $dst );
            } else {
                runtime_error( "$prefix: \"$dst\" already exists." );
            }; # if
        } else {
            runtime_error( "$prefix: \"$dst\" is not a file." );
        }; # if
    }; # if

    File::Copy::copy( $src, $dst ) or runtime_error( "$prefix: $!" );
    # On Windows* OS File::Copy preserves file attributes, but on Linux* OS it doesn't.
    # So we should do it manually...
    if ( $^O =~ m/^linux\z/ ) {
        my $mode = ( stat( $src ) )[ 2 ]
            or runtime_error( "$prefix: cannot get status info for source file." );
        chmod( $mode, $dst )
            or runtime_error( "$prefix: cannot change mode of destination file." );
    }; # if

}; # sub copy_file

# -------------------------------------------------------------------------------------------------

sub move_file($$@) {

    my $src  = shift( @_ );
    my $dst  = shift( @_ );
    my %opts = @_;
    my $prefix = "Could not move file \"$src\" to \"$dst\"";

    check_opts( %opts, [ qw( -overwrite ) ] );

    if ( not -e $src ) {
        runtime_error( "$prefix: \"$src\" does not exist." );
    }; # if
    if ( not -f $src ) {
        runtime_error( "$prefix: \"$src\" is not a file." );
    }; # if
    if ( -e $dst ) {
        if ( -f $dst ) {
            if ( $opts{ -overwrite } ) {
                #
            } else {
                runtime_error( "$prefix: \"$dst\" already exists." );
            }; # if
        } else {
            runtime_error( "$prefix: \"$dst\" is not a file." );
        }; # if
    }; # if

    File::Copy::move( $src, $dst ) or runtime_error( "$prefix: $!" );

}; # sub move_file

# -------------------------------------------------------------------------------------------------

sub del_file($) {
    my $files = shift( @_ );
    if ( ref( $files ) eq "" ) {
        $files = [ $files ];
    }; # if
    foreach my $file ( @$files ) {
        debug( "Deleting file `$file'..." );
        my $rc = unlink( $file );
        if ( $rc == 0 && $! != ENOENT ) {
            # Reporn an error, but ignore ENOENT, because the goal is achieved.
            runtime_error( "Deleting file `$file' failed: $!" );
        }; # if
    }; # foreach $file
}; # sub del_file

# -------------------------------------------------------------------------------------------------

=back

=cut

# =================================================================================================
# File I/O subroutines.
# =================================================================================================

=head2 File I/O subroutines.

=cut

#--------------------------------------------------------------------------------------------------

=head3 read_file

B<Synopsis:>

    read_file( $file, @options )

B<Description:>

Read file and return its content. In scalar context function returns a scalar, in list context
function returns list of lines.

Note: If the last of file does not terminate with newline, function will append it.

B<Arguments:>

=over

=item B<$file>

A name or handle of file to read from.

=back

B<Options:>

=over

=item B<-binary>

If true, file treats as a binary file: no newline conversion, no truncating trailing space, no
newline removing performed. Entire file returned as a scalar.

=item B<-bulk>

This option is allowed only in binary mode. Option's value should be a reference to a scalar.
If option present, file content placed to pointee scalar and function returns true (1).

=item B<-chomp>

If true, newline characters are removed from file content. By default newline characters remain.
This option is not applicable in binary mode.

=item B<-keep_trailing_space>

If true, trainling space remain at the ends of lines. By default all trailing spaces are removed.
This option is not applicable in binary mode.

=back

B<Examples:>

Return file as single line, remove trailing spaces.

    my $bulk = read_file( "message.txt" );

Return file as list of lines with removed trailing space and
newline characters.

    my @bulk = read_file( "message.txt", -chomp => 1 );

Read a binary file:

    my $bulk = read_file( "message.txt", -binary => 1 );

Read a big binary file:

    my $bulk;
    read_file( "big_binary_file", -binary => 1, -bulk => \$bulk );

Read from standard input:

    my @bulk = read_file( \*STDIN );

=cut

sub read_file($@) {

    my $file = shift( @_ );  # The name or handle of file to read from.
    my %opts = @_;           # Options.

    my $name;
    my $handle;
    my @bulk;
    my $error = \&runtime_error;

    my @binopts = qw( -binary -error -bulk );                       # Options available in binary mode.
    my @txtopts = qw( -binary -error -keep_trailing_space -chomp -layer ); # Options available in text (non-binary) mode.
    check_opts( %opts, [ @binopts, @txtopts ] );
    if ( $opts{ -binary } ) {
        check_opts( %opts, [ @binopts ], "these options cannot be used with -binary" );
    } else {
        check_opts( %opts, [ @txtopts ], "these options cannot be used without -binary" );
    }; # if
    if ( not exists( $opts{ -error } ) ) {
        $opts{ -error } = "error";
    }; # if
    if ( $opts{ -error } eq "warning" ) {
        $error = \&warning;
    } elsif( $opts{ -error } eq "ignore" ) {
        $error = sub {};
    } elsif ( ref( $opts{ -error } ) eq "ARRAY" ) {
        $error = sub { push( @{ $opts{ -error } }, $_[ 0 ] ); };
    }; # if

    if ( ( ref( $file ) eq "GLOB" ) or UNIVERSAL::isa( $file, "IO::Handle" ) ) {
        $name = "unknown";
        $handle = $file;
    } else {
        $name = $file;
        if ( get_ext( $file ) eq ".gz" and not $opts{ -binary } ) {
            $handle = IO::Zlib->new( $name, "rb" );
        } else {
            $handle = IO::File->new( $name, "r" );
        }; # if
        if ( not defined( $handle ) ) {
            $error->( "File \"$name\" could not be opened for input: $!" );
        }; # if
    }; # if
    if ( defined( $handle ) ) {
        if ( $opts{ -binary } ) {
            binmode( $handle );
            local $/ = undef;   # Set input record separator to undef to read entire file as one line.
            if ( exists( $opts{ -bulk } ) ) {
                ${ $opts{ -bulk } } = $handle->getline();
            } else {
                $bulk[ 0 ] = $handle->getline();
            }; # if
        } else {
            if ( defined( $opts{ -layer } ) ) {
                binmode( $handle, $opts{ -layer } );
            }; # if
            @bulk = $handle->getlines();
            # Special trick for UTF-8 files: Delete BOM, if any.
            if ( defined( $opts{ -layer } ) and $opts{ -layer } eq ":utf8" ) {
                if ( substr( $bulk[ 0 ], 0, 1 ) eq "\x{FEFF}" ) {
                    substr( $bulk[ 0 ], 0, 1 ) = "";
                }; # if
            }; # if
        }; # if
        $handle->close()
            or $error->( "File \"$name\" could not be closed after input: $!" );
    } else {
        if ( $opts{ -binary } and exists( $opts{ -bulk } ) ) {
            ${ $opts{ -bulk } } = "";
        }; # if
    }; # if
    if ( $opts{ -binary } ) {
        if ( exists( $opts{ -bulk } ) ) {
            return 1;
        } else {
            return $bulk[ 0 ];
        }; # if
    } else {
        if ( ( @bulk > 0 ) and ( substr( $bulk[ -1 ], -1, 1 ) ne "\n" ) ) {
            $bulk[ -1 ] .= "\n";
        }; # if
        if ( not $opts{ -keep_trailing_space } ) {
            map( $_ =~ s/\s+\n\z/\n/, @bulk );
        }; # if
        if ( $opts{ -chomp } ) {
            chomp( @bulk );
        }; # if
        if ( wantarray() ) {
            return @bulk;
        } else {
            return join( "", @bulk );
        }; # if
    }; # if

}; # sub read_file

#--------------------------------------------------------------------------------------------------

=head3 write_file

B<Synopsis:>

    write_file( $file, $bulk, @options )

B<Description:>

Write file.

B<Arguments:>

=over

=item B<$file>

The name or handle of file to write to.

=item B<$bulk>

Bulk to write to a file. Can be a scalar, or a reference to scalar or an array.

=back

B<Options:>

=over

=item B<-backup>

If true, create a backup copy of file overwritten. Backup copy is placed into the same directory.
The name of backup copy is the same as the name of file with `~' appended. By default backup copy
is not created.

=item B<-append>

If true, the text will be added to existing file.

=back

B<Examples:>

    write_file( "message.txt", \$bulk );
        # Write file, take content from a scalar.

    write_file( "message.txt", \@bulk, -backup => 1 );
        # Write file, take content from an array, create a backup copy.

=cut

sub write_file($$@) {

    my $file = shift( @_ );  # The name or handle of file to write to.
    my $bulk = shift( @_ );  # The text to write. Can be reference to array or scalar.
    my %opts = @_;           # Options.

    my $name;
    my $handle;

    check_opts( %opts, [ qw( -append -backup -binary -layer ) ] );

    my $mode = $opts{ -append } ? "a": "w";
    if ( ( ref( $file ) eq "GLOB" ) or UNIVERSAL::isa( $file, "IO::Handle" ) ) {
        $name = "unknown";
        $handle = $file;
    } else {
        $name = $file;
        if ( $opts{ -backup } and ( -f $name ) ) {
            copy_file( $name, $name . "~", -overwrite => 1 );
        }; # if
        $handle = IO::File->new( $name, $mode )
            or runtime_error( "File \"$name\" could not be opened for output: $!" );
    }; # if
    if ( $opts{ -binary } ) {
        binmode( $handle );
    } elsif ( $opts{ -layer } ) {
        binmode( $handle, $opts{ -layer } );
    }; # if
    if ( ref( $bulk ) eq "" ) {
        if ( defined( $bulk ) ) {
            $handle->print( $bulk );
            if ( not $opts{ -binary } and ( substr( $bulk, -1 ) ne "\n" ) ) {
                $handle->print( "\n" );
            }; # if
        }; # if
    } elsif ( ref( $bulk ) eq "SCALAR" ) {
        if ( defined( $$bulk ) ) {
            $handle->print( $$bulk );
            if ( not $opts{ -binary } and ( substr( $$bulk, -1 ) ne "\n" ) ) {
                $handle->print( "\n" );
            }; # if
        }; # if
    } elsif ( ref( $bulk ) eq "ARRAY" ) {
        foreach my $line ( @$bulk ) {
            if ( defined( $line ) ) {
                $handle->print( $line );
                if ( not $opts{ -binary } and ( substr( $line, -1 ) ne "\n" ) ) {
                    $handle->print( "\n" );
                }; # if
            }; # if
        }; # foreach
    } else {
        Carp::croak( "write_file: \$bulk must be a scalar or reference to (scalar or array)" );
    }; # if
    $handle->close()
        or runtime_error( "File \"$name\" could not be closed after output: $!" );

}; # sub write_file

#--------------------------------------------------------------------------------------------------

=cut

# =================================================================================================
# Execution subroutines.
# =================================================================================================

=head2 Execution subroutines.

=over

=cut

#--------------------------------------------------------------------------------------------------

sub _pre {

    my $arg = shift( @_ );

    # If redirection is not required, exit.
    if ( not exists( $arg->{ redir } ) ) {
        return 0;
    }; # if

    # Input parameters.
    my $mode   = $arg->{ mode   }; # Mode, "<" (input ) or ">" (output).
    my $handle = $arg->{ handle }; # Handle to manipulate.
    my $redir  = $arg->{ redir  }; # Data, a file name if a scalar, or file contents, if a reference.

    # Output parameters.
    my $save_handle;
    my $temp_handle;
    my $temp_name;

    # Save original handle (by duping it).
    $save_handle = Symbol::gensym();
    $handle->flush();
    open( $save_handle, $mode . "&" . $handle->fileno() )
        or die( "Cannot dup filehandle: $!" );

    # Prepare a file to IO.
    if ( UNIVERSAL::isa( $redir, "IO::Handle" ) or ( ref( $redir ) eq "GLOB" ) ) {
        # $redir is reference to an object of IO::Handle class (or its decedant).
        $temp_handle = $redir;
    } elsif ( ref( $redir ) ) {
        # $redir is a reference to content to be read/written.
        # Prepare temp file.
        ( $temp_handle, $temp_name ) =
            File::Temp::tempfile(
                "$tool.XXXXXXXX",
                DIR    => File::Spec->tmpdir(),
                SUFFIX => ".tmp",
                UNLINK => 1
            );
        if ( not defined( $temp_handle ) ) {
            runtime_error( "Could not create temp file." );
        }; # if
        if ( $mode eq "<" ) {
            # It is a file to be read by child, prepare file content to be read.
            $temp_handle->print( ref( $redir ) eq "SCALAR" ? ${ $redir } : @{ $redir } );
            $temp_handle->flush();
            seek( $temp_handle, 0, 0 );
                # Unfortunatelly, I could not use OO interface to seek.
                # ActivePerl 5.6.1 complains on both forms:
                #    $temp_handle->seek( 0 );    # As declared in IO::Seekable.
                #    $temp_handle->setpos( 0 );  # As described in documentation.
        } elsif ( $mode eq ">" ) {
            # It is a file for output. Clear output variable.
            if ( ref( $redir ) eq "SCALAR" ) {
                ${ $redir } = "";
            } else {
                @{ $redir } = ();
            }; # if
        }; # if
    } else {
        # $redir is a name of file to be read/written.
        # Just open file.
        if ( defined( $redir ) ) {
            $temp_name = $redir;
        } else {
            $temp_name = File::Spec->devnull();
        }; # if
        $temp_handle = IO::File->new( $temp_name, $mode )
            or runtime_error( "file \"$temp_name\" could not be opened for " . ( $mode eq "<" ? "input" : "output" ) . ": $!" );
    }; # if

    # Redirect handle to temp file.
    open( $handle, $mode . "&" . $temp_handle->fileno() )
        or die( "Cannot dup filehandle: $!" );

    # Save output parameters.
    $arg->{ save_handle } = $save_handle;
    $arg->{ temp_handle } = $temp_handle;
    $arg->{ temp_name   } = $temp_name;

}; # sub _pre


sub _post {

    my $arg = shift( @_ );

    # Input parameters.
    my $mode   = $arg->{ mode   }; # Mode, "<" or ">".
    my $handle = $arg->{ handle }; # Handle to save and set.
    my $redir  = $arg->{ redir  }; # Data, a file name if a scalar, or file contents, if a reference.

    # Parameters saved during preprocessing.
    my $save_handle = $arg->{ save_handle };
    my $temp_handle = $arg->{ temp_handle };
    my $temp_name   = $arg->{ temp_name   };

    # If no handle was saved, exit.
    if ( not $save_handle ) {
        return 0;
    }; # if

    # Close handle.
    $handle->close()
        or die( "$!" );

    # Read the content of temp file, if necessary, and close temp file.
    if ( ( $mode ne "<" ) and ref( $redir ) ) {
        $temp_handle->flush();
        seek( $temp_handle, 0, 0 );
        if ( $^O =~ m/MSWin/ ) {
            binmode( $temp_handle, ":crlf" );
        }; # if
        if ( ref( $redir ) eq "SCALAR" ) {
            ${ $redir } .= join( "", $temp_handle->getlines() );
        } elsif ( ref( $redir ) eq "ARRAY" ) {
            push( @{ $redir }, $temp_handle->getlines() );
        }; # if
    }; # if
    if ( not UNIVERSAL::isa( $redir, "IO::Handle" ) ) {
        $temp_handle->close()
            or die( "$!" );
    }; # if

    # Restore handle to original value.
    $save_handle->flush();
    open( $handle, $mode . "&" . $save_handle->fileno() )
        or die( "Cannot dup filehandle: $!" );

    # Close save handle.
    $save_handle->close()
        or die( "$!" );

    # Delete parameters saved during preprocessing.
    delete( $arg->{ save_handle } );
    delete( $arg->{ temp_handle } );
    delete( $arg->{ temp_name   } );

}; # sub _post

#--------------------------------------------------------------------------------------------------

=item C<execute( [ @command ], @options )>

Execute specified program or shell command.

Program is specified by reference to an array, that array is passed to C<system()> function which
executes the command. See L<perlfunc> for details how C<system()> interprets various forms of
C<@command>.

By default, in case of any error error message is issued and script terminated (by runtime_error()).
Function returns an exit code of program.

Alternatively, he function may return exit status of the program (see C<-ignore_status>) or signal
(see C<-ignore_signal>) so caller may analyze it and continue execution.

Options:

=over

=item C<-stdin>

Redirect stdin of program. The value of option can be:

=over

=item C<undef>

Stdin of child is attached to null device.

=item a string

Stdin of child is attached to a file with name specified by option.

=item a reference to a scalar

A dereferenced scalar is written to a temp file, and child's stdin is attached to that file.

=item a reference to an array

A dereferenced array is written to a temp file, and child's stdin is attached to that file.

=back

=item C<-stdout>

Redirect stdout. Possible values are the same as for C<-stdin> option. The only difference is
reference specifies a variable receiving program's output.

=item C<-stderr>

It similar to C<-stdout>, but redirects stderr. There is only one additional value:

=over

=item an empty string

means that stderr should be redirected to the same place where stdout is redirected to.

=back

=item C<-append>

Redirected stream will not overwrite previous content of file (or variable).
Note, that option affects both stdout and stderr.

=item C<-ignore_status>

By default, subroutine raises an error and exits the script if program returns non-exit status. If
this options is true, no error is raised. Instead, status is returned as function result (and $@ is
set to error message).

=item C<-ignore_signal>

By default, subroutine raises an error and exits the script if program die with signal. If
this options is true, no error is raised in such a case. Instead, signal number is returned (as
negative value), error message is placed to C<$@> variable.

If command is not even started, -256 is returned.

=back

Examples:

    execute( [ "cmd.exe", "/c", "dir" ] );
        # Execute NT shell with specified options, no redirections are
        # made.

    my $output;
    execute( [ "cvs", "-n", "-q", "update", "." ], -stdout => \$output );
        # Execute "cvs -n -q update ." command, output is saved
        # in $output variable.

    my @output;
    execute( [ qw( cvs -n -q update . ) ], -stdout => \@output, -stderr => undef );
        # Execute specified command,  output is saved in @output
        # variable, stderr stream is redirected to null device
        # (/dev/null in Linux* OS and nul in Windows* OS).

=cut

sub execute($@) {

    # !!! Add something to complain on unknown options...

    my $command = shift( @_ );
    my %opts    = @_;
    my $prefix  = "Could not execute $command->[ 0 ]";

    check_opts( %opts, [ qw( -stdin -stdout -stderr -append -ignore_status -ignore_signal ) ] );

    if ( ref( $command ) ne "ARRAY" ) {
        Carp::croak( "execute: $command must be a reference to array" );
    }; # if

    my $stdin  = { handle => \*STDIN,  mode => "<" };
    my $stdout = { handle => \*STDOUT, mode => ">" };
    my $stderr = { handle => \*STDERR, mode => ">" };
    my $streams = {
        stdin  => $stdin,
        stdout => $stdout,
        stderr => $stderr
    }; # $streams

    for my $stream ( qw( stdin stdout stderr ) ) {
        if ( exists( $opts{ "-$stream" } ) ) {
            if ( ref( $opts{ "-$stream" } ) !~ m/\A(|SCALAR|ARRAY)\z/ ) {
                Carp::croak( "execute: -$stream option: must have value of scalar, or reference to (scalar or array)." );
            }; # if
            $streams->{ $stream }->{ redir } = $opts{ "-$stream" };
        }; # if
        if ( $opts{ -append } and ( $streams->{ $stream }->{ mode } ) eq ">" ) {
            $streams->{ $stream }->{ mode } = ">>";
        }; # if
    }; # foreach $stream

    _pre( $stdin  );
    _pre( $stdout );
    if ( defined( $stderr->{ redir } ) and not ref( $stderr->{ redir } ) and ( $stderr->{ redir } eq "" ) ) {
        if ( exists( $stdout->{ redir } ) ) {
            $stderr->{ redir } = $stdout->{ temp_handle };
        } else {
            $stderr->{ redir } = ${ $stdout->{ handle } };
        }; # if
    }; # if
    _pre( $stderr );
    my $rc = system( @$command );
    my $errno = $!;
    my $child = $?;
    _post( $stderr );
    _post( $stdout );
    _post( $stdin  );

    my $exit = 0;
    my $signal_num  = $child & 127;
    my $exit_status = $child >> 8;
    $@ = "";

    if ( $rc == -1 ) {
        $@ = "\"$command->[ 0 ]\" failed: $errno";
        $exit = -256;
        if ( not $opts{ -ignore_signal } ) {
            runtime_error( $@ );
        }; # if
    } elsif ( $signal_num != 0 ) {
        $@ = "\"$command->[ 0 ]\" failed due to signal $signal_num.";
        $exit = - $signal_num;
        if ( not $opts{ -ignore_signal } ) {
            runtime_error( $@ );
        }; # if
    } elsif ( $exit_status != 0 ) {
        $@ = "\"$command->[ 0 ]\" returned non-zero status $exit_status.";
        $exit = $exit_status;
        if ( not $opts{ -ignore_status } ) {
            runtime_error( $@ );
        }; # if
    }; # if

    return $exit;

}; # sub execute

#--------------------------------------------------------------------------------------------------

=item C<backticks( [ @command ], @options )>

Run specified program or shell command and return output.

In scalar context entire output is returned in a single string. In list context list of strings
is returned. Function issues an error and exits script if any error occurs.

=cut


sub backticks($@) {

    my $command = shift( @_ );
    my %opts    = @_;
    my @output;

    check_opts( %opts, [ qw( -chomp ) ] );

    execute( $command, -stdout => \@output );

    if ( $opts{ -chomp } ) {
        chomp( @output );
    }; # if

    return ( wantarray() ? @output : join( "", @output ) );

}; # sub backticks

#--------------------------------------------------------------------------------------------------

sub pad($$$) {
    my ( $str, $length, $pad ) = @_;
    my $lstr = length( $str );    # Length of source string.
    if ( $lstr < $length ) {
        my $lpad  = length( $pad );                         # Length of pad.
        my $count = int( ( $length - $lstr ) / $lpad );     # Number of pad repetitions.
        my $tail  = $length - ( $lstr + $lpad * $count );
        $str = $str . ( $pad x $count ) . substr( $pad, 0, $tail );
    }; # if
    return $str;
}; # sub pad

# --------------------------------------------------------------------------------------------------

=back

=cut

#--------------------------------------------------------------------------------------------------

return 1;

#--------------------------------------------------------------------------------------------------

=cut

# End of file.
