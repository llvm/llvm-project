#
# This is not a runnable script, it is a Perl module, a collection of variables, subroutines, etc.
# to be used in Perl scripts.
#
# To get help about exported variables and subroutines, execute the following command:
#
#     perldoc Platform.pm
#
# or see POD (Plain Old Documentation) imbedded to the source...
#
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

package Platform;

use strict;
use warnings;

use base "Exporter";

use Uname;

my @vars;

BEGIN {
    @vars = qw{ $host_arch $host_os $host_platform $target_arch $target_mic_arch $target_os $target_platform };
}

our $VERSION     = "0.014";
our @EXPORT      = qw{};
our @EXPORT_OK   = ( qw{ canon_arch canon_os canon_mic_arch legal_arch arch_opt }, @vars );
our %EXPORT_TAGS = ( all => [ @EXPORT_OK ], vars => \@vars );

# Canonize architecture name.
sub canon_arch($) {
    my ( $arch ) = @_;
    if ( defined( $arch ) ) {
        if ( $arch =~ m{\A\s*(?:32|IA-?32|IA-?32 architecture|i[3456]86|x86)\s*\z}i ) {
            $arch = "32";
        } elsif ( $arch =~ m{\A\s*(?:48|(?:ia)?32e|Intel\s*64|Intel\(R\)\s*64|x86[_-]64|x64|AMD64)\s*\z}i ) {
            $arch = "32e";
        } elsif ( $arch =~ m{\Aarm(?:v7\D*)?\z} ) {
            $arch = "arm";
        } elsif ( $arch =~ m{\Appc64le} ) {
			$arch = "ppc64le";
        } elsif ( $arch =~ m{\Appc64} ) {
        	$arch = "ppc64";
        } elsif ( $arch =~ m{\Aaarch64} ) {
                $arch = "aarch64";
        } elsif ( $arch =~ m{\Amic} ) {
            $arch = "mic";
        } elsif ( $arch =~ m{\Amips64} ) {
            $arch = "mips64";
        } elsif ( $arch =~ m{\Amips} ) {
            $arch = "mips";
        } elsif ( $arch =~ m{\Ariscv64} ) {
            $arch = "riscv64";
        } elsif ( $arch =~ m{\Aloongarch64} ) {
            $arch = "loongarch64";
        } elsif ( $arch =~ m{\As390x} ) {
            $arch = "s390x";
        } else {
            $arch = undef;
        }; # if
    }; # if
    return $arch;
}; # sub canon_arch

# Canonize Intel(R) Many Integrated Core Architecture name.
sub canon_mic_arch($) {
    my ( $mic_arch ) = @_;
    if ( defined( $mic_arch ) ) {
        if ( $mic_arch =~ m{\Aknf} ) {
            $mic_arch = "knf";
        } elsif ( $mic_arch =~ m{\Aknc}) {
            $mic_arch = "knc";
        } elsif ( $mic_arch =~ m{\Aknl} ) {
            $mic_arch = "knl";
        } else {
            $mic_arch = undef;
        }; # if
    }; # if
    return $mic_arch;
}; # sub canon_mic_arch

{  # Return legal approved architecture name.
    my %legal = (
        "32"  => "IA-32 architecture",
        "32e" => "Intel(R) 64",
        "arm" => "ARM",
        "aarch64" => "AArch64",
        "loongarch64" => "LoongArch64",
        "mic" => "Intel(R) Many Integrated Core Architecture",
        "mips" => "MIPS",
        "mips64" => "MIPS64",
        "riscv64" => "RISC-V (64-bit)",
    );

    sub legal_arch($) {
        my ( $arch ) = @_;
        $arch = canon_arch( $arch );
        if ( defined( $arch ) ) {
            $arch = $legal{ $arch };
        }; # if
        return $arch;
    }; # sub legal_arch
}

{  # Return architecture name suitable for Intel compiler setup scripts.
    my %option = (
        "32"  => "ia32",
        "32e" => "intel64",
        "64"  => "ia64",
        "arm" => "arm",
        "aarch64" => "aarch",
        "mic" => "intel64",
        "mips" => "mips",
        "mips64" => "MIPS64",
    );

    sub arch_opt($) {
        my ( $arch ) = @_;
        $arch = canon_arch( $arch );
        if ( defined( $arch ) ) {
            $arch = $option{ $arch };
        }; # if
        return $arch;
    }; # sub arch_opt
}

# Canonize OS name.
sub canon_os($) {
    my ( $os ) = @_;
    if ( defined( $os ) ) {
        if ( $os =~ m{\A\s*(?:Linux|lin|l)\s*\z}i ) {
            $os = "lin";
        } elsif ( $os =~ m{\A\s*(?:Mac(?:\s*OS(?:\s*X)?)?|mac|m|Darwin)\s*\z}i ) {
            $os = "mac";
        } elsif ( $os =~ m{\A\s*(?:Win(?:dows)?(?:(?:_|\s*)?(?:NT|XP|95|98|2003))?|w)\s*\z}i ) {
            $os = "win";
        } else {
            $os = undef;
        }; # if
    }; # if
    return $os;
}; # sub canon_os

my ( $_host_os, $_host_arch, $_target_os, $_target_arch, $_target_mic_arch, $_default_mic_arch);

# Set the default mic-arch value.
$_default_mic_arch = "knc";

sub set_target_arch($) {
    my ( $arch ) = canon_arch( $_[ 0 ] );
    if ( defined( $arch ) ) {
        $_target_arch       = $arch;
        $ENV{ LIBOMP_ARCH } = $arch;
    }; # if
    return $arch;
}; # sub set_target_arch

sub set_target_mic_arch($) {
    my ( $mic_arch ) = canon_mic_arch( $_[ 0 ] );
    if ( defined( $mic_arch ) ) {
        $_target_mic_arch       = $mic_arch;
        $ENV{ LIBOMP_MIC_ARCH } = $mic_arch;
    }; # if
    return $mic_arch;
}; # sub set_target_mic_arch

sub set_target_os($) {
    my ( $os ) = canon_os( $_[ 0 ] );
    if ( defined( $os ) ) {
        $_target_os       = $os;
        $ENV{ LIBOMP_OS } = $os;
    }; # if
    return $os;
}; # sub set_target_os

sub target_options() {
    my @options = (
        "target-os|os=s" =>
            sub {
                set_target_os( $_[ 1 ] ) or
                    die "Bad value of --target-os option: \"$_[ 1 ]\"\n";
            },
        "target-architecture|target-arch|architecture|arch=s" =>
           sub {
               set_target_arch( $_[ 1 ] ) or
                   die "Bad value of --target-architecture option: \"$_[ 1 ]\"\n";
           },
        "target-mic-architecture|target-mic-arch|mic-architecture|mic-arch=s" =>
           sub {
               set_target_mic_arch( $_[ 1 ] ) or
                   die "Bad value of --target-mic-architecture option: \"$_[ 1 ]\"\n";
           },
    );
    return @options;
}; # sub target_options

# Detect host arch.
{
    my $hardware_platform = Uname::hardware_platform();
    if ( 0 ) {
    } elsif ( $hardware_platform eq "i386" ) {
        $_host_arch = "32";
    } elsif ( $hardware_platform eq "ia64" ) {
        $_host_arch = "64";
    } elsif ( $hardware_platform eq "x86_64" ) {
        $_host_arch = "32e";
    } elsif ( $hardware_platform eq "arm" ) {
        $_host_arch = "arm";
    } elsif ( $hardware_platform eq "ppc64le" ) {
        $_host_arch = "ppc64le";
    } elsif ( $hardware_platform eq "ppc64" ) {
        $_host_arch = "ppc64";
    } elsif ( $hardware_platform eq "aarch64" ) {
        $_host_arch = "aarch64";
    } elsif ( $hardware_platform eq "mips64" ) {
        $_host_arch = "mips64";
    } elsif ( $hardware_platform eq "mips" ) {
        $_host_arch = "mips";
    } elsif ( $hardware_platform eq "riscv64" ) {
        $_host_arch = "riscv64";
    } elsif ( $hardware_platform eq "loongarch64" ) {
        $_host_arch = "loongarch64";
    } elsif ( $hardware_platform eq "s390x" ) {
        $_host_arch = "s390x";
    } else {
        die "Unsupported host hardware platform: \"$hardware_platform\"; stopped";
    }; # if
}

# Detect host OS.
{
    my $operating_system = Uname::operating_system();
    if ( 0 ) {
    } elsif ( $operating_system eq "GNU/Linux" ) {
        $_host_os = "lin";
    } elsif ( $operating_system eq "FreeBSD" ) {
        # Host OS resembles Linux.
        $_host_os = "lin";
    } elsif ( $operating_system eq "NetBSD" ) {
        # Host OS resembles Linux.
        $_host_os = "lin";
    } elsif ( $operating_system eq "Darwin" ) {
        $_host_os = "mac";
    } elsif ( $operating_system eq "MS Windows" ) {
        $_host_os = "win";
    } else {
        die "Unsupported host operating system: \"$operating_system\"; stopped";
    }; # if
}

# Detect target arch.
if ( defined( $ENV{ LIBOMP_ARCH } ) ) {
    # Use arch specified in LIBOMP_ARCH.
    $_target_arch = canon_arch( $ENV{ LIBOMP_ARCH } );
    if ( not defined( $_target_arch ) ) {
        die "Unknown architecture specified in LIBOMP_ARCH environment variable: \"$ENV{ LIBOMP_ARCH }\"";
    }; # if
} else {
    # Otherwise use host architecture.
    $_target_arch = $_host_arch;
}; # if
$ENV{ LIBOMP_ARCH } = $_target_arch;

# Detect target Intel(R) Many Integrated Core Architecture.
if ( defined( $ENV{ LIBOMP_MIC_ARCH } ) ) {
    # Use mic arch specified in LIBOMP_MIC_ARCH.
    $_target_mic_arch = canon_mic_arch( $ENV{ LIBOMP_MIC_ARCH } );
    if ( not defined( $_target_mic_arch ) ) {
        die "Unknown architecture specified in LIBOMP_MIC_ARCH environment variable: \"$ENV{ LIBOMP_MIC_ARCH }\"";
    }; # if
} else {
    # Otherwise use default Intel(R) Many Integrated Core Architecture.
    $_target_mic_arch = $_default_mic_arch;
}; # if
$ENV{ LIBOMP_MIC_ARCH } = $_target_mic_arch;

# Detect target OS.
if ( defined( $ENV{ LIBOMP_OS } ) ) {
    # Use OS specified in LIBOMP_OS.
    $_target_os = canon_os( $ENV{ LIBOMP_OS } );
    if ( not defined( $_target_os ) ) {
        die "Unknown OS specified in LIBOMP_OS environment variable: \"$ENV{ LIBOMP_OS }\"";
    }; # if
} else {
    # Otherwise use host OS.
    $_target_os = $_host_os;
}; # if
$ENV{ LIBOMP_OS } = $_target_os;

use vars @vars;

tie( $host_arch,       "Platform::host_arch" );
tie( $host_os,         "Platform::host_os" );
tie( $host_platform,   "Platform::host_platform" );
tie( $target_arch,     "Platform::target_arch" );
tie( $target_mic_arch, "Platform::target_mic_arch" );
tie( $target_os,       "Platform::target_os" );
tie( $target_platform, "Platform::target_platform" );

{ package Platform::base;

    use Carp;

    use Tie::Scalar;
    use base "Tie::StdScalar";

    sub STORE {
        my $self = shift( @_ );
        croak( "Modifying \$" . ref( $self ) . " is not allowed; stopped" );
    }; # sub STORE

} # package Platform::base

{ package Platform::host_arch;
    use base "Platform::base";
    sub FETCH {
        return $_host_arch;
    }; # sub FETCH
} # package Platform::host_arch

{ package Platform::host_os;
    use base "Platform::base";
    sub FETCH {
        return $_host_os;
    }; # sub FETCH
} # package Platform::host_os

{ package Platform::host_platform;
    use base "Platform::base";
    sub FETCH {
        return "${_host_os}_${_host_arch}";
    }; # sub FETCH
} # package Platform::host_platform

{ package Platform::target_arch;
    use base "Platform::base";
    sub FETCH {
        return $_target_arch;
    }; # sub FETCH
} # package Platform::target_arch

{ package Platform::target_mic_arch;
    use base "Platform::base";
    sub FETCH {
        return $_target_mic_arch;
    }; # sub FETCH
} # package Platform::target_mic_arch

{ package Platform::target_os;
    use base "Platform::base";
    sub FETCH {
        return $_target_os;
    }; # sub FETCH
} # package Platform::target_os

{ package Platform::target_platform;
    use base "Platform::base";
    sub FETCH {
        if ($_target_arch eq "mic") {
            return "${_target_os}_${_target_mic_arch}";
        } else {
        return "${_target_os}_${_target_arch}";
        }
    }; # sub FETCH
} # package Platform::target_platform


return 1;

__END__

=pod

=head1 NAME

B<Platform.pm> -- Few subroutines to get OS, architecture and platform name in form suitable for
naming files, directories, macros, etc.

=head1 SYNOPSIS

    use Platform ":all";
    use tools;

    my $arch   = canon_arch( "em64T" );        # Returns "32e".
    my $legal  = legal_arch( "em64t" );        # Returns "Intel(R) 64".
    my $option = arch_opt( "em64t" );          # Returns "intel64".
    my $os     = canon_os( "Windows NT" );     # Returns "win".

    print( $host_arch, $host_os, $host_platform );
    print( $target_arch, $target_os, $target_platform );

    tools::get_options(
        Platform::target_options(),
        ...
    );


=head1 DESCRIPTION

Environment variable LIBOMP_OS specifies target OS to report. If LIBOMP_OS id not defined,
the script assumes host OS is target OS.

Environment variable LIBOMP_ARCH specifies target architecture to report. If LIBOMP_ARCH is not defined,
the script assumes host architecture is target one.

=head2 Functions.

=over

=item B<canon_arch( $arch )>

Input string is an architecture name to canonize. The function recognizes many variants, for example:
C<32e>, C<Intel64>, C<Intel(R) 64>, etc. Returned string is a canonized architecture name,
one of: C<32>, C<32e>, C<64>, C<arm>, C<ppc64le>, C<ppc64>, C<mic>, C<mips>, C<mips64>, C<riscv64>, C<loongarch64>, C<s390x>, or C<undef> is input string is not recognized.

=item B<legal_arch( $arch )>

Input string is architecture name. The function recognizes the same variants as C<arch_canon()> does.
Returned string is a name approved by Intel Legal, one of: C<IA-32 architecture>, C<Intel(R) 64>
or C<undef> if input string is not recognized.

=item B<arch_opt( $arch )>

Input string is architecture name. The function recognizes the same variants as C<arch_canon()> does.
Returned string is an architecture name suitable for passing to compiler setup scripts
(e. g. C<iccvars.sh>), one of: C<IA-32 architecture>, C<Intel(R) 64> or C<undef> if input string is not
recognized.

=item B<canon_os( $os )>

Input string is OS name to canonize. The function recognizes many variants, for example: C<mac>, C<OS X>, etc. Returned string is a canonized OS name, one of: C<lin>,
C<mac>, C<win>, or C<undef> is input string is not recognized.

=item B<target_options()>

Returns array suitable for passing to C<tools::get_options()> to let a script recognize
C<--target-architecture=I<str>> and C<--target-os=I<str>> options. Typical usage is:

    use tools;
    use Platform;

    my ( $os, $arch, $platform );    # Global variables, not initialized.

    ...

    get_options(
        Platform::target_options(),  # Let script recognize --target-os and --target-arch options.
        ...
    );
    # Initialize variables after parsing command line.
    ( $os, $arch, $platform ) = ( Platform::target_os(), Platform::target_arch(), Platform::target_platform() );

=back

=head2 Variables

=item B<$host_arch>

Canonized name of host architecture.

=item B<$host_os>

Canonized name of host OS.

=item B<$host_platform>

Host platform name (concatenated canonized OS name, underscore, and canonized architecture name).

=item B<$target_arch>

Canonized name of target architecture.

=item B<$target_os>

Canonized name of target OS.

=item B<$target_platform>

Target platform name (concatenated canonized OS name, underscore, and canonized architecture name).

=back

=cut

# end of file #
