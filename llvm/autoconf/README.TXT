Upgrading Libtool
===============================================================================

If you are in the mood to upgrade libtool, you must do the following:

 1. Get the new version of libtool and put it in <SRC>
 2. configure/build/install libtool with --prefix=<PFX>
 3. Copy <SRC>/ltdl.m4 to llvm/autoconf/m4
 4. Copy <PFX>/share/aclocal/libtool.m4 to llvm/autoconf/m4/libtool.m4
 5. Copy <PFX>/share/libtool/ltmain.sh to llvm/autoconf/ltmain.sh
 6. Copy <PFX>/share/libtool/libltdl/ltdl.c to llvm/lib/System
 7. Copy <PFX>/share/libtool/libltdl/ltdl.h to llvm/lib/System
 8. Edit the ltdl.h file to #include "llvm/Config/config.h" at the very top. You
    might also need to resolve some compiler warnings (typically about
    comparison of signed vs. unsigned values). But, you won't find out about 
    those until you build LLVM (step 13).
 9. Edit the llvm/autoconf/m4/libtool.m4 file so that:
    a) in AC_PROB_LIBTOOL macro, the value of LIBTOOL is set to
       $(top_builddir)/mklib, not $(top_builddir)/libtool
    b) in AC_LIBTOOL_SETUP macro, the variable default_ofile is set to 
       "mklib" instead of "libtool"
    c) s/AC_ENABLE_SHARED_DEFAULT/enable_shared_default/g
    d) s/AC_ENABLE_STATIC_DEFAULT/enable_static_default/g
    e) s/AC_ENABLE_FAST_INSTALL_DEFAULT/enable_fast_install_default/g
10. Run "autoupdate libtool.m4 ltdl.m4" in the llvm/autoconf/m4 directory.
    This should correctly update the macro definitions in the libtool m4 
    files to match the version of autoconf that LLVM uses. This converts
    AC_HELP_STRING to AS_HELP_STRING and AC_TRY_LINK to AC_LINK_IFELSE, amongst
    other things. You may need to manually adjust the files.
11. Run AutoRegen.sh to get the new macros into configure script
12. If there are any warnings from AutoRegen.sh, go to step 9.
13. Rebuild LLVM, making sure it reconfigures
14. Test the JIT which uses libltdl
15. If it all works, only THEN commit the changes.

Upgrading autoconf
===============================================================================

If you are in the mood to upgrade autoconf, you should:

 1. Consider not upgrading.
 2. No really, this is a hassle, you don't want to do it.
 3. Get the new version of autoconf and put it in <SRC>
 4. configure/build/install autoconf with --prefix=<PFX>
 5. Run autoupdate on all the m4 macros in llvm/autoconf/m4
 6. Run autoupdate on llvm/autoconf/configure.ac
 7. Regenerate configure script with AutoRegen.sh
 8. If there are any warnings from AutoRegen.sh, fix them and go to step 7.
 9. Test, test, test.
