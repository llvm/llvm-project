#
# This is not a runnable script, it is a Perl module, a collection of variables, subroutines, etc.
# To get help about exported variables and subroutines, execute the following command:
#
#     perldoc Uname.pm
#
# or see POD (Plain Old Documentation) embedded to the source...
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

package Uname;

use strict;
use warnings;
use warnings::register;
use Exporter;

use POSIX;
use File::Glob ":glob";
use Net::Domain qw{};

# Following code does not work with Perl 5.6 on Linux* OS and Windows* OS:
#
#     use if $^O eq "darwin", tools => qw{};
#
# The workaround for Perl 5.6:
#
BEGIN {
    if ( $^O eq "darwin" or $^O eq "linux" ) {
	require tools;
        import tools;
    }; # if
    if ( $^O eq "MSWin32" ) {
        require Win32;
    }; # if
}; # BEGIN

my $mswin = qr{\A(?:MSWin32|Windows_NT)\z};

my @posix = qw{ kernel_name fqdn kernel_release kernel_version machine };
    # Properties supported by POSIX::uname().
my @linux =
    qw{ processor hardware_platform operating_system };
    # Properties reported by uname in Linux* OS.
my @base = ( @posix, @linux );
    # Base properties.
my @aux =
    (
        qw{ host_name domain_name },
        map( "operating_system_$_", qw{ name release codename description } )
    );
    # Auxiliary properties.
my @all = ( @base, @aux );
    # All the properties.
my @meta = qw{ base_names all_names value };
    # Meta functions.

our $VERSION     = "0.07";
our @ISA         = qw{ Exporter };
our @EXPORT      = qw{};
our @EXPORT_OK   = ( @all, @meta );
our %EXPORT_TAGS =
    (
        base => [ @base ],
        all  => [ @all  ],
        meta => [ @meta ],
    );

my %values;
    # Hash of values. Some values are strings, some may be references to code which should be
    # evaluated to get real value. This trick is implemented because call to Net::Domain::hostfqdn()
    # is relatively slow.

# Get values from POSIX::uname().
@values{ @posix } = POSIX::uname();

# On some systems POSIX::uname() returns "short" node name (without domain name). To be consistent
# on all systems, we will get node name from alternative source.
if ( $^O =~ m/cygwin/i ) {
    # Function from Net::Domain module works well, but on Cygwin it prints to
    # stderr "domainname: not found". So we will use environment variables for now.
    $values{ fqdn } = lc( $ENV{ COMPUTERNAME } . "." . $ENV{ USERDNSDOMAIN } );
} else {
    # On systems other than Cygwin, let us use Net::Domain::hostfqdn(), but do it only node name
    # is really requested.
    $values{ fqdn } =
        sub {
            my $fqdn = Net::Domain::hostfqdn(); # "fqdn" stands for "fully qualified domain name".
            # On some systems POSIX::uname() and Net::Domain::hostfqdn() reports different names.
            # Let us issue a warning if they significantly different. Names are insignificantly
            # different if POSIX::uname() matches the beginning of Net::Domain::hostfqdn().
            if (
                $fqdn eq substr( $fqdn, 0, length( $fqdn ) )
                &&
                (
                    length( $fqdn ) == length( $fqdn )
                    ||
                    substr( $fqdn, length( $fqdn ), 1 ) eq "."
                )
            ) {
                # Ok.
            } else {
                warnings::warnif(
                    "POSIX::uname() and Net::Domain::hostfqdn() reported different names: " .
                        "\"$values{ fqdn }\" and \"$fqdn\" respectively\n"
                );
            }; # if
            return $fqdn;
        }; # sub
}; # if

if ( $^O =~ $mswin ) {
    if (
        $values{ machine } =~ m{\A(?:x86|[56]86)\z}
        and
        exists( $ENV{ PROCESSOR_ARCHITECTURE } ) and $ENV{ PROCESSOR_ARCHITECTURE } eq "x86"
        and
        exists( $ENV{ PROCESSOR_ARCHITEW6432 } )
    ) {
        if ( $ENV{ PROCESSOR_ARCHITEW6432 } eq "AMD64" ) {
            $values{ machine } = "x86_64";
        }; # if
    }; # if
}; # if

# Some values are not returned by POSIX::uname(), let us compute them.

# processor.
$values{ processor } = $values{ machine };

# hardware_platform.
if ( 0 ) {
} elsif ( $^O eq "linux" or $^O eq "freebsd" or $^O eq "netbsd" ) {
    if ( 0 ) {
    } elsif ( $values{ machine } =~ m{\Ai[3456]86\z} ) {
        $values{ hardware_platform } = "i386";
    } elsif ( $values{ machine } =~ m{\A(x86_64|amd64)\z} ) {
        $values{ hardware_platform } = "x86_64";
    } elsif ( $values{ machine } =~ m{\Aarmv7\D*\z} ) {
        $values{ hardware_platform } = "arm";
    } elsif ( $values{ machine } =~ m{\Appc64le\z} ) {
        $values{ hardware_platform } = "ppc64le";
    } elsif ( $values{ machine } =~ m{\Appc64\z} ) {
        $values{ hardware_platform } = "ppc64";
    } elsif ( $values{ machine } =~ m{\Aaarch64_32\z} ) {
        $values{ hardware_platform } = "aarch64_32";
    } elsif ( $values{ machine } =~ m{\Aaarch64\z} ) {
        $values{ hardware_platform } = "aarch64";
    } elsif ( $values{ machine } =~ m{\Amips64\z} ) {
        $values{ hardware_platform } = "mips64";
    } elsif ( $values{ machine } =~ m{\Amips\z} ) {
        $values{ hardware_platform } = "mips";
    } elsif ( $values{ machine } =~ m{\Ariscv64\z} ) {
        $values{ hardware_platform } = "riscv64";
    } elsif ( $values{ machine } =~ m{\Aloongarch64\z} ) {
        $values{ hardware_platform } = "loongarch64";
    } elsif ( $values{ machine } =~ m{\As390x\z} ) {
        $values{ hardware_platform } = "s390x";
    } else {
        die "Unsupported machine (\"$values{ machine }\") returned by POSIX::uname(); stopped";
    }; # if
} elsif ( $^O eq "darwin" ) {
    if ( 0 ) {
    } elsif ( $values{ machine } eq "x86" or $values{ machine } eq "i386" ) {
        $values{ hardware_platform } =
            sub {
                my $platform = "i386";
                # Some OSes on Intel(R) 64 still reports "i386" machine. Verify it by using
                # the value returned by 'sysctl -n hw.optional.x86_64'. On Intel(R) 64-bit systems the
                # value == 1; on 32-bit systems the 'hw.optional.x86_64' property either does not exist
                # or the value == 0. The path variable does not contain a path to sysctl when
                # started by crontab.
                my $sysctl = ( which( "sysctl" ) or "/usr/sbin/sysctl" );
                my $output;
                debug( "Executing $sysctl..." );
                execute( [ $sysctl, "-n", "hw.optional.x86_64" ], -stdout => \$output, -stderr => undef );
                chomp( $output );
                if ( 0 ) {
                } elsif ( "$output" eq "" or "$output" eq "0" ) {
                    $platform = "i386";
                } elsif ( "$output" eq "1" ) {
                    $platform = "x86_64";
                } else {
                    die "Unsupported value (\"$output\") returned by \"$sysctl -n hw.optional.x86_64\"; stopped";
                }; # if
                return $platform;
            }; # sub {
    } elsif ( $values{ machine } eq "x86_64" ) {
	# Some OS X* versions report "x86_64".
	$values{ hardware_platform } = "x86_64";
    } else {
        die "Unsupported machine (\"$values{ machine }\") returned by POSIX::uname(); stopped";
    }; # if
} elsif ( $^O =~ $mswin ) {
    if ( 0 ) {
    } elsif ( $values{ machine } =~ m{\A(?:x86|[56]86)\z} ) {
        $values{ hardware_platform } = "i386";
    } elsif ( $values{ machine } eq "x86_64" or $values{ machine } eq "amd64" ) {
        # ActivePerl for IA-32 architecture returns "x86_64", while ActivePerl for Intel(R) 64 returns "amd64".
        $values{ hardware_platform } = "x86_64";
    } else {
        die "Unsupported machine (\"$values{ machine }\") returned by POSIX::uname(); stopped";
    }; # if
} elsif ( $^O eq "cygwin" ) {
    if ( 0 ) {
    } elsif ( $values{ machine } =~ m{\Ai[3456]86\z} ) {
        $values{ hardware_platform } = "i386";
    } elsif ( $values{ machine } eq "x86_64" ) {
        $values{ hardware_platform } = "x86_64";
    } else {
        die "Unsupported machine (\"$values{ machine }\") returned by POSIX::uname(); stopped";
    }; # if
} else {
    die "Unsupported OS (\"$^O\"); stopped";
}; # if

# operating_system.
if ( 0 ) {
} elsif ( $values{ kernel_name } eq "Linux" ) {
    $values{ operating_system } = "GNU/Linux";
    my $release;    # Name of chosen "*-release" file.
    my $bulk;       # Content of release file.
    # On Ubuntu, lsb-release is quite informative, e. g.:
    #     DISTRIB_ID=Ubuntu
    #     DISTRIB_RELEASE=9.04
    #     DISTRIB_CODENAME=jaunty
    #     DISTRIB_DESCRIPTION="Ubuntu 9.04"
    # Try lsb-release first. But on some older systems lsb-release is not informative.
    # It may contain just one line:
    #     LSB_VERSION="1.3"
    $release = "/etc/lsb-release";
    if ( -e $release ) {
        $bulk = read_file( $release );
    } else {
        $bulk = "";
    }; # if
    if ( $bulk =~ m{^DISTRIB_} ) {
        # Ok, this lsb-release is informative.
        $bulk =~ m{^DISTRIB_ID\s*=\s*(.*?)\s*$}m
            or runtime_error( "$release: There is no DISTRIB_ID:", $bulk, "(eof)" );
        $values{ operating_system_name } = $1;
        $bulk =~ m{^DISTRIB_RELEASE\s*=\s*(.*?)\s*$}m
            or runtime_error( "$release: There is no DISTRIB_RELEASE:", $bulk, "(eof)" );
        $values{ operating_system_release } = $1;
        $bulk =~ m{^DISTRIB_CODENAME\s*=\s*(.*?)\s*$}m
            or runtime_error( "$release: There is no DISTRIB_CODENAME:", $bulk, "(eof)" );
        $values{ operating_system_codename } = $1;
        $bulk =~ m{^DISTRIB_DESCRIPTION\s*="?\s*(.*?)"?\s*$}m
            or runtime_error( "$release: There is no DISTRIB_DESCRIPTION:", $bulk, "(eof)" );
        $values{ operating_system_description } = $1;
    } else {
        # Oops. lsb-release is missed or not informative. Try other *-release files.
        $release = "/etc/system-release";
        if ( not -e $release ) {    # Use /etc/system-release" if such file exists.
            # Otherwise try other "/etc/*-release" files, but ignore "/etc/lsb-release".
            my @releases = grep( $_ ne "/etc/lsb-release", bsd_glob( "/etc/*-release" ) );
            # On some Fedora systems there are two files: fedora-release and redhat-release
            # with identical content. If fedora-release present, ignore redjat-release.
            if ( grep( $_ eq "/etc/fedora-release", @releases ) ) {
                @releases = grep( $_ ne "/etc/redhat-release", @releases );
            }; # if
            if ( @releases == 1 ) {
                $release = $releases[ 0 ];
            } else {
                if ( @releases == 0 ) {
                    # No *-release files found, try debian_version.
                    $release = "/etc/debian_version";
                    if ( not -e $release ) {
                        $release = undef;
                        warning( "No release files found in \"/etc/\" directory." );
                    }; # if
                } else {
                    $release = undef;
                    warning( "More than one release files found in \"/etc/\" directory:", @releases );
                }; # if
            }; # if
        }; # if
        if ( defined( $release ) ) {
            $bulk = read_file( $release );
            if ( $release =~ m{system|redhat|fedora} ) {
                # Red Hat or Fedora. Parse the first line of file.
                # Typical values of *-release (one of):
                #     Red Hat Enterprise Linux* OS Server release 5.2 (Tikanga)
                #     Red Hat Enterprise Linux* OS AS release 3 (Taroon Update 4)
                #     Fedora release 10 (Cambridge)
                $bulk =~ m{\A(.*)$}m
                    or runtime_error( "$release: Cannot find the first line:", $bulk, "(eof)" );
                my $first_line = $1;
                $values{ operating_system_description } = $first_line;
                $first_line =~ m{\A(.*?)\s+release\s+(.*?)(?:\s+\((.*?)(?:\s+Update\s+(.*?))?\))?\s*$}
                    or runtime_error( "$release:1: Cannot parse line:", $first_line );
                $values{ operating_system_name    }  = $1;
                $values{ operating_system_release }  = $2 . ( defined( $4 ) ? ".$4" : "" );
                $values{ operating_system_codename } = $3;
            } elsif ( $release =~ m{SuSE} ) {
                # Typical SuSE-release:
                #     SUSE Linux* OS Enterprise Server 10 (x86_64)
                #     VERSION = 10
                #     PATCHLEVEL = 2
                $bulk =~ m{\A(.*)$}m
                    or runtime_error( "$release: Cannot find the first line:", $bulk, "(eof)" );
                my $first_line = $1;
                $values{ operating_system_description } = $first_line;
                $first_line =~ m{^(.*?)\s*(\d+)\s*\(.*?\)\s*$}
                    or runtime_error( "$release:1: Cannot parse line:", $first_line );
                $values{ operating_system_name } = $1;
                $bulk =~ m{^VERSION\s*=\s*(.*)\s*$}m
                    or runtime_error( "$release: There is no VERSION:", $bulk, "(eof)" );
                $values{ operating_system_release } = $1;
                if ( $bulk =~ m{^PATCHLEVEL\s*=\s*(.*)\s*$}m ) {
                    $values{ operating_system_release } .= ".$1";
                }; # if
            } elsif ( $release =~ m{debian_version} ) {
                # Debian. The file debian_version contains just version number, nothing more:
                #     4.0
                my $name = "Debian";
                $bulk =~ m{\A(.*)$}m
                    or runtime_error( "$release: Cannot find the first line:", $bulk, "(eof)" );
                my $version = $1;
                $values{ operating_system_name        } = $name;
                $values{ operating_system_release     } = $version;
                $values{ operating_system_codename    } = "unknown";
                $values{ operating_system_description } = sprintf( "%s %s", $name, $version );
            }; # if
        }; # if
    }; # if
    if ( not defined( $values{ operating_system_name } ) ) {
        $values{ operating_system_name } = "GNU/Linux";
    }; # if
} elsif ( $values{ kernel_name } eq "Darwin" ) {
    my %codenames = (
        10.4 => "Tiger",
        10.5 => "Leopard",
        10.6 => "Snow Leopard",
    );
   my $darwin;
   my $get_os_info =
       sub {
           my ( $name ) = @_;
           if ( not defined $darwin ) {
               $darwin->{ operating_system } = "Darwin";
               # sw_vers prints OS X* version to stdout:
               #     ProductName:       OS X*
               #     ProductVersion:    10.4.11
               #     BuildVersion:      8S2167
               # It does not print codename, so we code OS X* codenames here.
               my $sw_vers = which( "sw_vers" ) || "/usr/bin/sw_vers";
               my $output;
               debug( "Executing $sw_vers..." );
               execute( [ $sw_vers ], -stdout => \$output, -stderr => undef );
               $output =~ m{^ProductName:\s*(.*)\s*$}m
                   or runtime_error( "There is no ProductName in sw_vers output:", $output, "(eof)" );
               my $name = $1;
               $output =~ m{^ProductVersion:\s*(.*)\s*$}m
                   or runtime_error( "There is no ProductVersion in sw_vers output:", $output, "(eof)" );
               my $release = $1;
               # Sometimes release reported as "10.4.11" (3 components), sometimes as "10.6".
               # Handle both variants.
               $release =~ m{^(\d+.\d+)(?:\.\d+)?(?=\s|$)}
                   or runtime_error( "Cannot parse OS X* version: $release" );
               my $version = $1;
               my $codename = ( $codenames{ $version } or "unknown" );
               $darwin->{ operating_system_name        } = $name;
               $darwin->{ operating_system_release     } = $release;
               $darwin->{ operating_system_codename    } = $codename;
               $darwin->{ operating_system_description } = sprintf( "%s %s (%s)", $name, $release, $codename );
           }; # if
           return $darwin->{ $name };
       }; # sub
    $values{ operating_system             } = sub { $get_os_info->( "operating_system"             ); };
    $values{ operating_system_name        } = sub { $get_os_info->( "operating_system_name"        ); };
    $values{ operating_system_release     } = sub { $get_os_info->( "operating_system_release"     ); };
    $values{ operating_system_codename    } = sub { $get_os_info->( "operating_system_codename"    ); };
    $values{ operating_system_description } = sub { $get_os_info->( "operating_system_description" ); };
} elsif ( $values{ kernel_name } =~ m{\AWindows[ _]NT\z} ) {
    $values{ operating_system } = "MS Windows";
    # my @os_name = Win32::GetOSName();
    # $values{ operating_system_release } = $os_name[ 0 ];
    # $values{ operating_system_update  } = $os_name[ 1 ];
} elsif ( $values{ kernel_name } =~ m{\ACYGWIN_NT-} ) {
    $values{ operating_system } = "MS Windows";
} elsif ( $values{ kernel_name } =~ m{\AFreeBSD} ) {
    $values{ operating_system } = "FreeBSD";
} elsif ( $values{ kernel_name } =~ m{\ANetBSD} ) {
    $values{ operating_system } = "NetBSD";
} else {
    die "Unsupported kernel_name (\"$values{ kernel_name }\") returned by POSIX::uname(); stopped";
}; # if

# host_name and domain_name
$values{ host_name } =
    sub {
        my $fqdn = value( "fqdn" );
        $fqdn =~ m{\A([^.]*)(?:\.(.*))?\z};
        my $host_name = $1;
        if ( not defined( $host_name ) or $host_name eq "" ) {
            die "Unexpected error: undefined or empty host name; stopped";
        }; # if
        return $host_name;
    };
$values{ domain_name } =
    sub {
        my $fqdn = value( "fqdn" );
        $fqdn =~ m{\A([^.]*)(?:\.(.*))?\z};
        my $domain_name = $2;
        if ( not defined( $domain_name ) or $domain_name eq "" ) {
            die "Unexpected error: undefined or empty domain name; stopped";
        }; # if
        return $domain_name;
    };

# Replace undefined values with "unknown".
foreach my $name ( @all ) {
    if ( not defined( $values{ $name } ) ) {
        $values{ $name } = "unknown";
    }; # if
}; # foreach $name

# Export functions reporting properties.
foreach my $name ( @all ) {
    no strict "refs";
    *$name = sub { return value( $name ); };
}; # foreach $name

# This function returns base names.
sub base_names {
    return @base;
}; # sub base_names

# This function returns all the names.
sub all_names {
    return @all;
}; # sub all_names

# This function returns value by the specified name.
sub value($) {
    my $name = shift( @_ );
    if ( ref( $values{ $name } ) ) {
        my $value = $values{ $name }->();
        $values{ $name } = $value;
    }; # if
    return $values{ $name };
}; # sub value

return 1;

__END__

=pod

=head1 NAME

B<Uname.pm> -- A few subroutines to get system information usually provided by
C</bin/uname> and C<POSIX::uname()>.

=head1 SYNOPSIS

    use Uname;

    # Base property functions.
    $kernel_name       = Uname::kernel_name();
    $fqdn              = Uname::fqdn();
    $kernel_release    = Uname::kernel_release();
    $kernel_version    = Uname::kernel_version();
    $machine           = Uname::machine();
    $processor         = Uname::processor();
    $hardware_platform = Uname::hardware_platform();
    $operating_system  = Uname::operating_system();

    # Auxiliary property functions.
    $host_name         = Uname::host_name();
    $domain_name       = Uname::domain_name();
    $os_name           = Uname::operating_system_name();
    $os_release        = Uname::operating_system_release();
    $os_codename       = Uname::operating_system_codename();
    $os_description    = Uname::operating_system_description();

    # Meta functions.
    @base_names  = Uname::base_names();
    @all_names   = Uname::all_names();
    $kernel_name = Uname::value( "kernel_name" );

=head1 DESCRIPTION

B<Uname.pm> resembles functionality found in C<POSIX::uname()> function or in C<uname> program.
However, both C<POSIX::uname()> and C</bin/uname> have some disadvantages:

=over

=item *

C<uname> may be not available in some environments, for example, in Windows* OS
(C<uname> may be found in some third-party software packages, like MKS Toolkit or Cygwin, but it is
not a part of OS).

=item *

There are many different versions of C<uname>. For example, C<uname> on OS X* does not
recognize options C<-i>, C<-o>, and any long options.

=item *

Different versions of C<uname> may report the same property differently. For example,
C<uname> on Linux* OS reports machine as C<i686>, while C<uname> on OS X* reports the same machine as
C<x86>.

=item *

C<POSIX::uname()> returns list of values. I cannot recall what is the fourth element of the list.

=back

=head2 Base Functions

Base property functions provide the information as C<uname> program.

=over

=item B<kernel_name()>

Returns the kernel name, as reported by C<POSIX::uname()>.

=item B<fqdn()>

Returns the FQDN, fully qualified domain name. On some systems C<POSIX::uname()> reports short node
name (with no domain name), on others C<POSIX::uname()> reports full node name. This
function strive to return FQDN always (by refining C<POSIX::uname()> with
C<Net::Domain::hostfqdn()>).

=item B<kernel_release()>

Returns the kernel release string, as reported by C<POSIX::uname()>. Usually the string consists of
several numbers, separated by dots and dashes, but may also include some non-numeric substrings like
"smp".

=item B<kernel_version()>

Returns the kernel version string, as reported by C<POSIX::uname()>. It is B<not> several
dot-separated numbers but much longer string describing the kernel.
For example, on Linux* OS it includes build date.
If you look for something identifying the kernel, look at L<kernel_release>.

=item B<machine()>

Returns the machine hardware name, as reported by POSIX::uname(). Not reliable. Different OSes may
report the same machine hardware name differently. For example, Linux* OS reports C<i686>, while OS X*
reports C<x86> on the same machine.

=item B<processor()>

Returns the processor type. Not reliable. Usually the same as C<machine>.

=item B<hardware_platform()>

One of: C<i386> or C<x86_64>.

=item B<operating_system()>

One of: C<GNU/Linux>, C<OS X*>, or C<MS Windows>.

=back

=head2 Auxiliary Functions

Auxiliary functions extends base functions with information not reported by C<uname> program.

Auxiliary functions collect information from different sources. For example, on OS X*, they may
call C<sw_vers> program to find out OS release; on Linux* OS they may parse C</etc/redhat-release> file,
etc.

=over

=item B<host_name()>

Returns host name (FQDN with dropped domain part).

=item B<domain_name()>

Returns domain name (FQDN with dropped host part).

=item B<operating_system_name>

Name of operating system or name of Linux* OS distribution, like "Fedora" or
"Red Hat Enterprise Linux* OS Server".

=item B<operating_system_release>

Release (version) of operating system or Linux* OS distribution. Usually it is a series of
dot-separated numbers.

=item B<operating_system_codename>

Codename of operating system release or Linux* OS distribution. For example, Fedora 10 is "Cambridge"
while OS X* 10.4 is "Tiger".

=item B<operating_system_description>

Longer string. Usually it includes all the operating system properting mentioned above -- name,
release, codename in parentheses.

=back

=head2 Meta Functions

=over

=item B<base_names()>

This function returns the list of base property names.

=item B<all_names()>

This function returns the list of all property names.

=item B<value(> I<name> B<)>

This function returns the value of the property specified by I<name>.

=back

=head1 EXAMPLES

    use Uname;

    print( Uname::string(), "\n" );

    foreach my $name ( Uname::all_names() ) {
        print( "$name=\"" . Uname::value( $name ) . "\"\n" );
    }; # foreach $name

=head1 SEE ALSO

L<POSIX::uname>, L<uname>.

=cut

# end of file #

