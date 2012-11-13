#!/bin/sh
die () {
	echo "$@" 1>&2
	exit 1
}

test -d autoconf && test -f autoconf/configure.ac && cd autoconf
test -f configure.ac || die "Can't find 'autoconf' dir; please cd into it first"
autoconf --version | egrep '2\.[5-6][0-9]' > /dev/null
if test $? -ne 0 ; then
	die "Your autoconf was not detected as being 2.5x"
fi
cwd=`pwd`
if test -d ../../../autoconf/m4 ; then
  cd ../../../autoconf/m4
  llvm_m4=`pwd`
  cd $cwd
elif test -d ../../llvm/autoconf/m4 ; then
  cd ../../llvm/autoconf/m4
  llvm_m4=`pwd`
  cd $cwd
else
  die "Can't find the LLVM autoconf/m4 directory. polly should be checked out to projects directory"
fi
echo "Regenerating aclocal.m4 with aclocal"
rm -f aclocal.m4
aclocal -I $llvm_m4 -I "$llvm_m4/.." -I $(pwd)/m4 || die "aclocal failed"
echo "Regenerating configure with autoconf 2.5x"
autoconf --force --warnings=all -o ../configure configure.ac || die "autoconf failed"
cd ..
echo "Regenerating config.h.in with autoheader"
autoheader --warnings=all -I autoconf -I autoconf/m4 -I $llvm_m4 -I "$llvm_m4/.." autoconf/configure.ac || die "autoheader failed"
exit 0
