#
#//===----------------------------------------------------------------------===//
#//
#//                     The LLVM Compiler Infrastructure
#//
#// This file is dual licensed under the MIT and the University of Illinois Open
#// Source Licenses. See LICENSE.txt for details.
#//
#//===----------------------------------------------------------------------===//
#
package LibOMP;

use strict;
use warnings;

use tools;

sub empty($) {
    my ( $var ) = @_;
    return (not exists($ENV{$var})) or (not defined($ENV{$var})) or ($ENV{$var} eq "");
}; # sub empty

my ( $base, $out, $tmp );
if ( empty( "LIBOMP_WORK" ) ) {
    # $FindBin::Bin is not used intentionally because it gives real path. I want to use absolute,
    # but not real one (real path does not contain symlinks while absolute path may contain
    # symlinks).
    $base = get_dir( get_dir( abs_path( $0 ) ) );
} else {
    $base = abs_path( $ENV{ LIBOMP_WORK } );
}; # if

if ( empty( "LIBOMP_EXPORTS" ) ) {
    $out = cat_dir( $base, "exports" );
} else {
    $out = abs_path( $ENV{ LIBOMP_EXPORTS } );
}; # if

if ( empty( "LIBOMP_TMP" ) ) {
    $tmp = cat_dir( $base, "tmp" );
} else {
    $tmp = abs_path( $ENV{ LIBOMP_TMP } );
}; # if

$ENV{ LIBOMP_WORK    } = $base;
$ENV{ LIBOMP_EXPORTS } = $out;
$ENV{ LIBOMP_TMP     } = $tmp;

return 1;

__END__

=pod

=head1 NAME

B<LibOMP.pm> --

=head1 SYNOPSIS

    use FindBin;
    use lib "$FindBin::Bin/lib";
    use LibOMP;

    $ENV{ LIBOMP_WORK    }
    $ENV{ LIBOMP_TMP     }
    $ENV{ LIBOMP_EXPORTS }

=head1 DESCRIPTION

The module checks C<LIBOMP_WORK>, C<LIBOMP_EXPORTS>, and C<LIBOMP_TMP> environments variables.
If a variable set, the module makes sure it is absolute. If a variable does not exist, the module
sets it to default value.

Default value for C<LIBOMP_EXPORTS> is C<$LIBOMP_WORK/exports>, for C<LIBOMP_TMP> --
C<$LIBOMP_WORK/tmp>.

Value for C<LIBOMP_WORK> is guessed. The module assumes the script (which uses the module) is
located in C<tools/> directory of libomp directory tree, and uses path of the script to calculate
C<LIBOMP_WORK>,

=cut

# end of file #

