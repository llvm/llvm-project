#
# Provide the arguments and other processing needed for an LLVM project
#
AC_DEFUN([LLVM_CONFIG_PROJECT],
  [AC_ARG_WITH([llvmsrc],
    AS_HELP_STRING([--with-llvmsrc],[Location of LLVM Source Code]),
    [llvm_src="$withval"],[llvm_src="]$1["])
  AC_SUBST(LLVM_SRC,$llvm_src)
  AC_ARG_WITH([llvmobj],
    AS_HELP_STRING([--with-llvmobj],[Location of LLVM Object Code]),
    [llvm_obj="$withval"],[llvm_obj="]$2["])
  AC_SUBST(LLVM_OBJ,$llvm_obj)
  AC_CONFIG_COMMANDS([setup],,[llvm_src="${LLVM_SRC}"])
])
