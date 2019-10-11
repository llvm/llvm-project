#!/usr/bin/env perl

#
#//===----------------------------------------------------------------------===//
#//
#// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#// See https://llvm.org/LICENSE.txt for license information.
#// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#//
#//===----------------------------------------------------------------------===//
#

use strict;
use warnings;

use FindBin;
use lib "$FindBin::Bin/lib";

use tools;

our $VERSION = "0.005";
my $target_os;
my $target_arch;

# --------------------------------------------------------------------------------------------------
# Ouput parse error.
#     $tool -- Name of tool.
#     @bulk -- Output of the tool.
#     $n    -- Number of line caused parse error.
sub parse_error($\@$) {
    my ( $tool, $bulk, $n ) = @_;
    my @bulk;
    for ( my $i = 0; $i < @$bulk; ++ $i ) {
        push( @bulk, ( $i == $n ? ">>> " : "    " ) . $bulk->[ $i ] );
    }; # for $i
    runtime_error( "Fail to parse $tool output:", @bulk, "(eof)" );
}; # sub parse_error


# --------------------------------------------------------------------------------------------------
# Linux* OS version of get_deps() parses output of ldd:
#
# $ ldd libname.so
#   libc.so.6 => /lib64/libc.so.6 (0x00002b60fedd8000)
#   libdl.so.2 => /lib64/libdl.so.2 (0x00002b60ff12b000)
#   libpthread.so.0 => /lib64/libpthread.so.0 (0x00002b60ff32f000)
#   /lib64/ld-linux-x86-64.so.2 (0x0000003879400000)
#
# Note: ldd printd all the dependencies, direct and indirect. (For example, if specified library
# requires libdl.so, and libdl.so requires /lib/ld-linux.so, ldd prints both libdl.so and
# /lib/ld-linux.so). If you do not want indirect dependencies, look at readelf tool.
#
sub get_deps_ldd($) {

    my $lib = shift ( @_ );
    my $tool = "ldd";
    my @bulk;
    my @deps;

    execute( [ $tool, $lib ], -stdout => \@bulk );
    debug( @bulk, "(eof)" );

    foreach my $i ( 0 .. @bulk - 1 ) {
        my $line = $bulk[ $i ];
        if ( $line !~ m{^\s*(?:([_a-z0-9.+-/]*)\s+=>\s+)?([_a-z0-9.+-/]*)\s+\(0x[0-9a-z]*\)$}i ) {
            parse_error( $tool, @bulk, $i );
        }; # if
        my $dep = ( defined( $1 ) ? $1 : $2 );
        push( @deps, $dep );
    }; # foreach $i

    return @deps;

}; # sub get_deps_ldd


# --------------------------------------------------------------------------------------------------
# Another Linux* OS version of get_deps() parses output of readelf:
#
# $ readelf -d exports/lin_32e/lib/libomp.so
#
# Dynamic segment at offset 0x87008 contains 24 entries:
#   Tag        Type                         Name/Value
#  0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
#  0x0000000000000001 (NEEDED)             Shared library: [libdl.so.2]
#  0x0000000000000001 (NEEDED)             Shared library: [libpthread.so.0]
#  0x000000000000000e (SONAME)             Library soname: [libomp.so]
#  0x000000000000000d (FINI)               0x51caa
#  0x0000000000000004 (HASH)               0x158
#  0x0000000000000005 (STRTAB)             0x9350
#  ...
#
# Note: In contrast to ldd, readlef shows only direct dependencies.
#
sub get_deps_readelf($) {

    my $file = shift ( @_ );
    my $tool;
    my @bulk;
    my @deps;

    if($target_arch eq "mic") {
        $tool = "x86_64-k1om-linux-readelf";
    } else {
        $tool = "readelf";
    }

    # Force the readelf call to be in English. For example, when readelf
    # is called on a french localization, it will find "Librairie partagees"
    # instead of shared library
    $ENV{ LANG } = "C";

    execute( [ $tool, "-d", $file ], -stdout => \@bulk );
    debug( @bulk, "(eof)" );

    my $i = 0;
    # Parse header.
    ( $i < @bulk and $bulk[ $i ] =~ m{^\s*$} )
        or parse_error( $tool, @bulk, $i );
    ++ $i;
    if ( $i == @bulk - 1 and $bulk[ $i ] =~ m{^There is no dynamic section in this file\.\s*$} ) {
        # This is not dynamic executable => no dependencies.
        return @deps;
    }; # if
    ( $i < @bulk and $bulk[ $i ] =~ m{^Dynamic (?:segment|section) at offset 0x[0-9a-f]+ contains \d+ entries:\s*$} )
        or parse_error( $tool, @bulk, $i );
    ++ $i;
    ( $i < @bulk and $bulk[ $i ] =~ m{^\s*Tag\s+Type\s+Name/Value\s*$} )
        or parse_error( $tool, @bulk, $i );
    ++ $i;
    # Parse body.
    while ( $i < @bulk ) {
        my $line = $bulk[ $i ];
        if ( $line !~ m{^\s*0x[0-9a-f]+\s+\(?([_A-Z0-9]+)\)?\s+(.*)\s*$}i ) {
            parse_error( $tool, @bulk, $i );
        }; # if
        my ( $type, $value ) = ( $1, $2 );
        if ( $type eq "NEEDED" ) {
            if ( $value !~ m{\AShared library: \[(.*)\]\z} ) {
                parse_error( $tool, @bulk, $i );
            }; # if
            my $dep = $1;
            push( @deps, $dep );
        }; # if
        ++ $i;
    }; # foreach $i

    return @deps;

}; # sub get_deps_readelf


# --------------------------------------------------------------------------------------------------
# OS X* version of get_deps() parses output of otool:
#
# $ otool -L libname.dylib
# exports/mac_32/lib.thin/libomp.dylib:
#        libomp.dylib (compatibility version 5.0.0, current version 5.0.0)
#        /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 88.1.3)
#
sub get_deps_otool($) {

    my $file = shift ( @_ );
    my $name = get_file( $file );
    my $tool = "otool";
    my @bulk;
    my @deps;

    if ( $target_arch eq "32e" ) {
        # On older (Tiger) systems otool does not recognize 64-bit binaries, so try to locate
        # otool64.
        my $path = which( "otool64" );
        if ( defined ( $path ) ) {
            $tool = "otool64";
        }; # if
    }; # if

    execute( [ $tool, "-L", $file ], -stdout => \@bulk );
    debug( @bulk, "(eof)" );

    my $i = 0;
    # Parse the first one or two lines separately.
    ( $i < @bulk and $bulk[ $i ] =~ m{^\Q$file\E:$} )
        or parse_error( $tool, @bulk, $i );
    ++ $i;
    if ( $name =~ m{\.dylib\z} ) {
        # Added "@rpath/" enables dynamic load of the library designated at link time.
        $name = '@rpath/' . $name;
        # In case of dynamic library otool print the library itself as a dependent library.
        ( $i < @bulk and $bulk[ $i ] =~ m{^\s+\Q$name\E\s+\(compatibility version.*\)$} )
            or parse_error( $tool, @bulk, $i );
        ++ $i;
    }; # if

    # Then parse the rest.
    while ( $i < @bulk ) {
        my $line = $bulk[ $i ];
        if ( $line !~ m/^\s*(.*)\s+\(compatibility version\s.*\)$/ ) {
            parse_error( $tool, @bulk, $i );
        }; # if
        my ( $dep ) = ( $1 );
        push( @deps, $dep );
        ++ $i;
    }; # while

    return @deps;

}; # sub get_deps_otool


# --------------------------------------------------------------------------------------------------
# Windows* OS version of get_deps() parses output of link:
#
# > link -dump -dependents libname.dll
# Microsoft (R) COFF/PE Dumper Version 8.00.40310.39
# Copyright (C) Microsoft Corporation.  All rights reserved.
# Dump of file S:\Projects.OMP\users\omalyshe\omp\libomp\exports\win_64\lib\libompmd.dll
# File Type: DLL
#   Image has the following dependencies:
#     KERNEL32.dll
#   Summary
#         C000 .data
#         6000 .pdata
#        18000 .rdata
#        ...
#
# > link -dump -directives libname.lib
# Microsoft (R) COFF/PE Dumper Version 8.00.40310.39
# Copyright (C) Microsoft Corporation.  All rights reserved.
# Dump of file S:\Projects.OMP\users\omalyshe\omp\libomp\exports\win_32e\lib\libimp5mt.lib
# File Type: LIBRARY
#   Linker Directives
#   -----------------
#   -defaultlib:"uuid.lib"
#   -defaultlib:"uuid.lib"
#   .....
#   Summary
#       3250 .bss
#       3FBC .data
#         34 .data1
#       ....
sub get_deps_link($) {

    my ( $lib ) = @_;
    my $tool = "link";
    my @bulk;
    my @deps;

    my $ext = lc( get_ext( $lib ) );
    if ( $ext !~ m{\A\.(?:lib|dll|exe)\z}i ) {
        runtime_error( "Incorrect file is specified: `$lib'; only `lib', `dll' or `exe' file expected" );
    }; # if

    execute(
        [ $tool, "/dump", ( $ext eq ".lib" ? "/directives" : "/dependents" ), $lib ],
        -stdout => \@bulk
    );

    debug( @bulk, "(eof)" );

    my $i = 0;
    ( $i < @bulk and $bulk[ $i ] =~ m{^Microsoft \(R\) COFF\/PE Dumper Version.*$} ) or parse_error( $tool, @bulk, $i ); ++ $i;
    ( $i < @bulk and $bulk[ $i ] =~ m{^Copyright \(C\) Microsoft Corporation\..*$} ) or parse_error( $tool, @bulk, $i ); ++ $i;
    ( $i < @bulk and $bulk[ $i ] =~ m{^\s*$}                                       ) or parse_error( $tool, @bulk, $i ); ++ $i;
    ( $i < @bulk and $bulk[ $i ] =~ m{^\s*$}                                       ) or parse_error( $tool, @bulk, $i ); ++ $i;
    ( $i < @bulk and $bulk[ $i ] =~ m{^Dump of file\s\Q$lib\E$}                    ) or parse_error( $tool, @bulk, $i ); ++ $i;
    ( $i < @bulk and $bulk[ $i ] =~ m{^\s*$}                                       ) or parse_error( $tool, @bulk, $i ); ++ $i;
    ( $i < @bulk and $bulk[ $i ] =~ m{^File Type:\s(.*)$}                          ) or parse_error( $tool, @bulk, $i ); ++ $i;
    ( $i < @bulk and $bulk[ $i ] =~ m{^\s*$}                                       ) or parse_error( $tool, @bulk, $i ); ++ $i;

    if ( $ext eq ".lib" ) {

        my %deps;
        while ( $i < @bulk ) {
            my $line = $bulk[ $i ];
            if ( 0 ) {
            } elsif ( $line =~ m{^\s*[-/]defaultlib\:(.*)\s*$}i ) {
                my $dep = $1;
                # Normalize library name:
                $dep = lc( $1 );              # Convert to lower case.
                $dep =~ s{\A"(.*)"\z}{$1};    # Drop surrounding quotes (if any).
                $dep =~ s{\.lib\z}{};         # Drop .lib suffix (if any).
                $deps{ $dep } = 1;
            } elsif ( $line =~ m{^\s*Linker Directives\s*$} ) {
            } elsif ( $line =~ m{^\s*-+\s*$} ) {
            } elsif ( $line =~ m{^\s*/alternatename\:.*$} ) {
            } elsif ( $line =~ m{^\s*$} ) {
            } elsif ( $line =~ m{^\s*/FAILIFMISMATCH\:.*$} ) {
                # This directive is produced only by _MSC_VER=1600
            } elsif ( $line =~ m{^\s*Summary\s*$} ) {
                last;
            } else {
                parse_error( $tool, @bulk, $i );
            }; # if
            ++ $i;
        } # while
        @deps = keys( %deps );

    } else {

        ( $i < @bulk and $bulk[ $i ] =~ m{\s*Image has the following dependencies\:$} )
            or parse_error( $tool, @bulk, $i );
        ++ $i;
        while ( $i < @bulk ) {
            my $line = $bulk[ $i ];
            if ( 0 ) {
            } elsif ( $line =~ m{^\s*$} ) {
                # Ignore empty lines.
            } elsif ( $line =~ m{^\s*(.*\.dll)$}i ) {
                my $dep = lc( $1 );
                push( @deps, $dep );
            } elsif ( $line =~ m{^\s*Summary$} ) {
                last;
            } else {
                parse_error( $tool, @bulk, $i );
            }; # if
            ++ $i;
        }; # while

    }; # if

    return @deps;

}; # sub get_deps_link


# --------------------------------------------------------------------------------------------------
# Main.
# --------------------------------------------------------------------------------------------------

# Parse command line.
my $expected;
my $bare;
Getopt::Long::Configure( "permute" );
get_options(
    "os=s"       => \$target_os,
    "arch=s"     => \$target_arch,
    "bare"       => \$bare,
    "expected=s" => \$expected,
);
my @expected;
if ( defined( $expected ) ) {
    if ( $expected ne "none" ) {
        @expected = sort( split( ",", $expected ) );
        if ( $target_os eq "win" ) {
            @expected = map( lc( $_ ), @expected );
        }; # if
    }; # if
}; # if
if ( @ARGV < 1 ) {
    cmdline_error( "Specify a library name to check for dependencies" );
}; # if
if ( @ARGV > 1 ) {
    cmdline_error( "Too many arguments" );
}; # if
my $lib = shift( @ARGV );
if ( not -e $lib ){
    runtime_error( "Specified file does not exist: \"$lib\"" );
}; # if

# Select appropriate get_deps implementation.
if ( 0 ) {
} elsif ( $target_os eq "lin" ) {
    *get_deps = \*get_deps_readelf;
} elsif ( $target_os eq "mac" ) {
    *get_deps = \*get_deps_otool;
} elsif ( $target_os eq "win" ) {
    *get_deps = \*get_deps_link;
} else {
    runtime_error( "OS \"$target_os\" not supported" );
}; # if

# Do the work.
my @deps = sort( get_deps( $lib ) );
if ( $bare ) {
    print( map( "$_\n", @deps ) );
} else {
    info( "Dependencies:", @deps ? map( "    $_", @deps ) : "(none)" );
}; # if
if ( defined( $expected ) ) {
    my %deps = map( ( $_ => 1 ), @deps );
    foreach my $dep ( @expected ) {
        delete( $deps{ $dep } );
    }; # foreach
    my @unexpected = sort( keys( %deps ) );
    if ( @unexpected ) {
        runtime_error( "Unexpected dependencies:", map( "    $_", @unexpected ) );
    }; # if
}; # if

exit( 0 );

__END__

=pod

=head1 NAME

B<check-depends.pl> -- Check dependencies for a specified library.

=head1 SYNOPSIS

B<check-depends.pl> I<OPTIONS>... I<library>

=head1 DESCRIPTION

C<check-depends.pl> finds direct dependencies for a specified library. List of actual dependencies
is sorted alphabetically and printed. If list of expected dependencies is specified, the scripts
checks the library has only allowed dependencies. In case of not expected depndencies the script
issues error message and exits with non-zero code.

Linux* OS and OS X*: The script finds dependencies only for dymamic libraries. Windows* OS: The script
finds dependencies for either static or dymamic libraries.

The script uses external tools. On Linux* OS, it runs F<readelf>, on OS X* -- F<otool> (or F<otool64>),
on Windows* OS -- F<link>.

On Windows* OS dependencies are printed in lower case, case of expected dependencies ignored.

=head1 OPTIONS

=over

=item B<--bare>

Do not use fancy formatting; produce plain, bare output: just a list of libraries,
a library per line.

=item B<--expected=>I<list>

I<list> is comma-separated list of expected dependencies (or C<none>).
If C<--expected> option specified, C<check-depends.pl> checks the specified library
has only expected dependencies.

=item B<--os=>I<str>

Specify target OS (tool to use) manually.
Useful for cross-build, when host OS is not the same as target OS.
I<str> should be either C<lin>, C<mac>, or C<win>.

=back

=head2 Standard Options

=over

=item B<--help>

Print short help message and exit.

=item B<--doc>

=item B<--manual>

Print full documentation and exit.

=item B<--quiet>

Do not output informational messages.

=item B<--version>

Print version and exit.

=back

=head1 ARGUMENTS

=over

=item I<library>

A name of library to find or check dependencies.

=back

=head1 EXAMPLES

Just print library dependencies (Windows* OS):

    > check-depends.pl exports/win_32/lib/libompmd.dll
    check-depends.pl: (i) Dependencies:
    check-depends.pl: (i)     kernel32.dll

Print library dependencies, use bare output (Linux* OS):

    $ check-depends.pl --bare exports/lin_32e/lib/libomp_db.so
    libc.so.6
    libdl.so.2
    libpthread.so.0

Check the library does not have any dependencies (OS X*):

    $ check-depends.pl --expected=none exports/mac_32/lib/libomp.dylib
    check-depends.pl: (i) Dependencies:
    check-depends.pl: (i)     /usr/lib/libSystem.B.dylib
    check-depends.pl: (x) Unexpected dependencies:
    check-depends.pl: (x)     /usr/lib/libSystem.B.dylib
    $ echo $?
    2

=cut

# end of file #

