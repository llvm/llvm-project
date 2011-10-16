#!/bin/bash

die() {
  echo "$@" 1>&2
  exit 1
}

clean() {
  echo $1 | sed -e 's/\\//g'
}

### NOTE: ############################################################
### These variables specify the tool versions we want to use.
### Periods should be escaped with backslash for use by grep.
###
### If you update these, please also update docs/GettingStarted.html
want_autoconf_version='2\.61'
want_autoheader_version=$want_autoconf_version
want_aclocal_version='1\.10'
want_libtool_version='1\.5\.22'
### END NOTE #########################################################

outfile=configure
configfile=configure.ac

want_autoconf_version_clean=$(clean $want_autoconf_version)
want_autoheader_version_clean=$(clean $want_autoheader_version)
want_aclocal_version_clean=$(clean $want_aclocal_version)
want_libtool_version_clean=$(clean $want_libtool_version)

test -d autoconf && test -f autoconf/$configfile && cd autoconf
test -f $configfile || die "Can't find 'autoconf' dir; please cd into it first"
autoconf --version | grep $want_autoconf_version > /dev/null
test $? -eq 0 || die "Your autoconf was not detected as being $want_autoconf_version_clean"
aclocal --version | grep '^aclocal.*'$want_aclocal_version > /dev/null
test $? -eq 0 || die "Your aclocal was not detected as being $want_aclocal_version_clean"
autoheader --version | grep '^autoheader.*'$want_autoheader_version > /dev/null
test $? -eq 0 || die "Your autoheader was not detected as being $want_autoheader_version_clean"
libtool --version | grep $want_libtool_version > /dev/null
test $? -eq 0 || die "Your libtool was not detected as being $want_libtool_version_clean"
echo ""
echo "### NOTE: ############################################################"
echo "### If you get *any* warnings from autoconf below you MUST fix the"
echo "### scripts in the m4 directory because there are future forward"
echo "### compatibility or platform support issues at risk. Please do NOT"
echo "### commit any configure script that was generated with warnings"
echo "### present. You should get just three 'Regenerating..' lines."
echo "######################################################################"
echo ""
echo "Regenerating aclocal.m4 with aclocal $want_aclocal_version_clean"
cwd=`pwd`
aclocal --force -I $cwd/m4 || die "aclocal failed"
echo "Regenerating configure with autoconf $want_autoconf_version_clean"
autoconf --force --warnings=all -o ../$outfile $configfile || die "autoconf failed"
cd ..
echo "Regenerating config.h.in with autoheader $want_autoheader_version_clean"
autoheader --warnings=all -I autoconf -I autoconf/m4 autoconf/$configfile || die "autoheader failed"
exit 0
