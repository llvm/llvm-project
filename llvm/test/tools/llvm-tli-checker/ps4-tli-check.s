# REQUIRES: x86-registered-target
#
## The object files ps4-tli-checks.right.so and ps4-tli-checks.wrong.so
## were generated with the following commands:
## llvm-mc --triple=x86_64-scei-ps4 --filetype=obj ps4-tli-check.s -o t.o
## ld.lld --shared t.o -o Inputs/ps4-tli-check.right.so
## llvm-mc --triple=x86_64-scei-ps4 --defsym WRONG=1 --filetype=obj ps4-tli-check.s -o t2.o
## ld.lld --shared t2.o -o Inputs/ps4-tli-check.wrong.so
#
# RUN: llvm-tli-checker --triple=x86_64-scei-ps4 %S/Inputs/ps4-tli-check.right.so | FileCheck %s
#
# RUN: echo %S/Inputs/ps4-tli-check.wrong.so > %t2.txt
# RUN: llvm-tli-checker --triple x86_64-scei-ps4 @%t2.txt | \
# RUN:     FileCheck %s --check-prefix=WRONG_SUMMARY --check-prefix=WRONG_DETAIL \
# RUN:     --implicit-check-not="==" --implicit-check-not="<<" --implicit-check-not=">>"
# RUN: llvm-tli-checker --triple x86_64-scei-ps4 @%t2.txt --report=summary | \
# RUN:     FileCheck %s --check-prefix=WRONG_SUMMARY \
# RUN:     --implicit-check-not="==" --implicit-check-not="<<" --implicit-check-not=">>"
## --separate implies --report=summary.
# RUN: llvm-tli-checker --triple x86_64-scei-ps4 @%t2.txt --separate | \
# RUN:     FileCheck %s --check-prefix=WRONG_SUMMARY \
# RUN:     --implicit-check-not="==" --implicit-check-not="<<" --implicit-check-not=">>"
#
# RUN: llvm-tli-checker --triple x86_64-scei-ps4 --dump-tli > %t3.txt
# RUN: FileCheck %s --check-prefix=AVAIL --input-file %t3.txt
# RUN: FileCheck %s --check-prefix=UNAVAIL --input-file %t3.txt
#
# CHECK: << Total TLI yes SDK no:  0
# CHECK: >> Total TLI no  SDK yes: 0
# CHECK: == Total TLI yes SDK yes: 235
#
# WRONG_DETAIL: << TLI yes SDK no : '_ZdaPv'
# WRONG_DETAIL: >> TLI no  SDK yes: '_ZdaPvj'
# WRONG_SUMMARY: << Total TLI yes SDK no:  1{{$}}
# WRONG_SUMMARY: >> Total TLI no  SDK yes: 1{{$}}
# WRONG_SUMMARY: == Total TLI yes SDK yes: 234
#
## The -COUNT suffix doesn't care if there are too many matches, so check
## the exact count first; the two directives should add up to that.
# AVAIL: TLI knows 466 symbols, 235 available
# AVAIL-COUNT-235: {{^}} available
# UNAVAIL-COUNT-231: not available

.macro defname name
.globl \name
.type  \name ,@function
\name : nop
.endm

.text
# For the WRONG case, omit _ZdaPv and include _ZdaPvj.
.ifdef WRONG
defname _ZdaPvj
.else
defname _ZdaPv
.endif
defname _ZdaPvRKSt9nothrow_t
defname _ZdaPvSt11align_val_t
defname _ZdaPvSt11align_val_tRKSt9nothrow_t
defname _ZdaPvm
defname _ZdaPvmSt11align_val_t
defname _ZdlPv
defname _ZdlPvRKSt9nothrow_t
defname _ZdlPvSt11align_val_t
defname _ZdlPvSt11align_val_tRKSt9nothrow_t
defname _ZdlPvm
defname _ZdlPvmSt11align_val_t
defname _Znam
defname _ZnamRKSt9nothrow_t
defname _ZnamSt11align_val_t
defname _ZnamSt11align_val_tRKSt9nothrow_t
defname _Znwm
defname _ZnwmRKSt9nothrow_t
defname _ZnwmSt11align_val_t
defname _ZnwmSt11align_val_tRKSt9nothrow_t
defname __cxa_atexit
defname __cxa_guard_abort
defname __cxa_guard_acquire
defname __cxa_guard_release
defname abs
defname acos
defname acosf
defname acosh
defname acoshf
defname acoshl
defname acosl
defname aligned_alloc
defname asin
defname asinf
defname asinh
defname asinhf
defname asinhl
defname asinl
defname atan
defname atan2
defname atan2f
defname atan2l
defname atanf
defname atanh
defname atanhf
defname atanhl
defname atanl
defname atof
defname atoi
defname atol
defname atoll
defname calloc
defname cbrt
defname cbrtf
defname cbrtl
defname ceil
defname ceilf
defname ceill
defname clearerr
defname copysign
defname copysignf
defname copysignl
defname cos
defname cosf
defname cosh
defname coshf
defname coshl
defname cosl
defname exp
defname exp2
defname exp2f
defname exp2l
defname expf
defname expl
defname expm1
defname expm1f
defname expm1l
defname fabs
defname fabsf
defname fabsl
defname fclose
defname fdopen
defname feof
defname ferror
defname fflush
defname fgetc
defname fgetpos
defname fgets
defname fileno
defname floor
defname floorf
defname floorl
defname fmax
defname fmaxf
defname fmaxl
defname fmin
defname fminf
defname fminl
defname fmod
defname fmodf
defname fmodl
defname fopen
defname fprintf
defname fputc
defname fputs
defname fread
defname free
defname frexp
defname frexpf
defname frexpl
defname fscanf
defname fseek
defname fsetpos
defname ftell
defname fwrite
defname getc
defname getchar
defname gets
defname isdigit
defname labs
defname ldexp
defname ldexpf
defname ldexpl
defname llabs
defname log
defname log10
defname log10f
defname log10l
defname log1p
defname log1pf
defname log1pl
defname log2
defname log2f
defname log2l
defname logb
defname logbf
defname logbl
defname logf
defname logl
defname malloc
defname memalign
defname memchr
defname memcmp
defname memcpy
defname memmove
defname memset
defname mktime
defname modf
defname modff
defname modfl
defname nearbyint
defname nearbyintf
defname nearbyintl
defname perror
defname posix_memalign
defname pow
defname powf
defname powl
defname printf
defname putc
defname putchar
defname puts
defname qsort
defname realloc
defname remainder
defname remainderf
defname remainderl
defname remove
defname rewind
defname rint
defname rintf
defname rintl
defname round
defname roundf
defname roundl
defname scanf
defname setbuf
defname setvbuf
defname sin
defname sinf
defname sinh
defname sinhf
defname sinhl
defname sinl
defname snprintf
defname sprintf
defname sqrt
defname sqrtf
defname sqrtl
defname sscanf
defname strcasecmp
defname strcat
defname strchr
defname strcmp
defname strcoll
defname strcpy
defname strcspn
defname strdup
defname strlen
defname strncasecmp
defname strncat
defname strncmp
defname strncpy
defname strpbrk
defname strrchr
defname strspn
defname strstr
defname strtod
defname strtof
defname strtok
defname strtok_r
defname strtol
defname strtold
defname strtoll
defname strtoul
defname strtoull
defname strxfrm
defname tan
defname tanf
defname tanh
defname tanhf
defname tanhl
defname tanl
defname trunc
defname truncf
defname truncl
defname ungetc
defname vfprintf
defname vfscanf
defname vprintf
defname vscanf
defname vsnprintf
defname vsprintf
defname vsscanf
defname wcslen

