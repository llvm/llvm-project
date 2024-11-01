# RUN: not llvm-mc -triple riscv64 -mattr=+experimental-zfa,+d,+zfh < %s 2>&1 | FileCheck -check-prefixes=CHECK-NO-RV32 %s
# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-zfa,+d,+zfh < %s 2>&1 | FileCheck -check-prefixes=CHECK-NO-RV64 %s

# Invalid rounding modes
# CHECK-NO-RV64: error: operand must be 'rtz' floating-point rounding mode
# CHECK-NO-RV32: error: operand must be 'rtz' floating-point rounding mode
fcvtmod.w.d a1, ft1, rne

# CHECK-NO-RV64: error: operand must be 'rtz' floating-point rounding mode
# CHECK-NO-RV32: error: operand must be 'rtz' floating-point rounding mode
fcvtmod.w.d a1, ft1, dyn

# CHECK-NO-RV64: error: operand must be 'rtz' floating-point rounding mode
# CHECK-NO-RV32: error: operand must be 'rtz' floating-point rounding mode
fcvtmod.w.d a1, ft1, rmm

# CHECK-NO-RV64: error: operand must be 'rtz' floating-point rounding mode
# CHECK-NO-RV32: error: operand must be 'rtz' floating-point rounding mode
fcvtmod.w.d a1, ft1, rdn

# CHECK-NO-RV64: error: operand must be 'rtz' floating-point rounding mode
# CHECK-NO-RV32: error: operand must be 'rtz' floating-point rounding mode
fcvtmod.w.d a1, ft1, rup

# Invalid floating-point immediate
# CHECK-NO-RV64: error: operand must be a valid floating-point constant
# CHECK-NO-RV32: error: operand must be a valid floating-point constant
fli.s ft1, 5.250000e-01

# CHECK-NO-RV64: error: operand must be a valid floating-point constant
# CHECK-NO-RV32: error: operand must be a valid floating-point constant
fli.d ft1, 3.560000e+02

# CHECK-NO-RV64: error: operand must be a valid floating-point constant
# CHECK-NO-RV32: error: operand must be a valid floating-point constant
fli.h ft1, 1.600000e+00

# CHECK-NO-RV64: error: invalid floating point immediate
# CHECK-NO-RV32: error: invalid floating point immediate
fli.s ft1, -min

# CHECK-NO-RV64: error: invalid floating point immediate
# CHECK-NO-RV32: error: invalid floating point immediate
fli.s ft1, -inf

# CHECK-NO-RV64: error: invalid floating point immediate
# CHECK-NO-RV32: error: invalid floating point immediate
fli.s ft1, -nan

# Don't accept decimal minimum.
# CHECK-NO-RV64: error: operand must be a valid floating-point constant
# CHECK-NO-RV32: error: operand must be a valid floating-point constant
fli.s ft1, 1.1754943508222875079687365372222456778186655567720875215087517062784172594547271728515625e-38

# Don't accept decimal minimum.
# CHECK-NO-RV64: error: operand must be a valid floating-point constant
# CHECK-NO-RV32: error: operand must be a valid floating-point constant
fli.d ft1, 2.225073858507201383090232717332404064219215980462331830553327416887204434813918195854283159012511020564067339731035811005152434161553460108856012385377718821130777993532002330479610147442583636071921565046942503734208375250806650616658158948720491179968591639648500635908770118304874799780887753749949451580451605050915399856582470818645113537935804992115981085766051992433352114352390148795699609591288891602992641511063466313393663477586513029371762047325631781485664350872122828637642044846811407613911477062801689853244110024161447421618567166150540154285084716752901903161322778896729707373123334086988983175067838846926092773977972858659654941091369095406136467568702398678315290680984617210924625396728515625e-308

# Don't accept decimal minimum.
# CHECK-NO-RV64: error: operand must be a valid floating-point constant
# CHECK-NO-RV32: error: operand must be a valid floating-point constant
fli.h ft1, 6.103516e-05

# Don't accept single precision minimum for double.
# CHECK-NO-RV64: error: operand must be a valid floating-point constant
# CHECK-NO-RV32: error: operand must be a valid floating-point constant
fli.d ft1, 1.1754943508222875079687365372222456778186655567720875215087517062784172594547271728515625e-38

# Don't accept single precision minimum for half.
# CHECK-NO-RV64: error: operand must be a valid floating-point constant
# CHECK-NO-RV32: error: operand must be a valid floating-point constant
fli.h ft1, 1.1754943508222875079687365372222456778186655567720875215087517062784172594547271728515625e-38

# Don't accept integers.
# CHECK-NO-RV32: error: invalid floating point immediate
# CHECK-NO-RV64: error: invalid floating point immediate
fli.s ft1, 1
