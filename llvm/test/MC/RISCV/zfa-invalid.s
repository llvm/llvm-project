# RUN: llvm-mc --verify=no-rv32 -triple riscv64 -mattr=+zfa,+q,+zfh < %s
# RUN: llvm-mc --verify=no-rv64 -triple riscv32 -mattr=+zfa,+q,+zfh < %s

# Invalid rounding modes
fcvtmod.w.d a1, ft1, rne
# no-rv64-error@-1 {{operand must be 'rtz' floating-point rounding mode}}
# no-rv32-error@-2 {{operand must be 'rtz' floating-point rounding mode}}

fcvtmod.w.d a1, ft1, dyn
# no-rv64-error@-1 {{operand must be 'rtz' floating-point rounding mode}}
# no-rv32-error@-2 {{operand must be 'rtz' floating-point rounding mode}}

fcvtmod.w.d a1, ft1, rmm
# no-rv64-error@-1 {{operand must be 'rtz' floating-point rounding mode}}
# no-rv32-error@-2 {{operand must be 'rtz' floating-point rounding mode}}

fcvtmod.w.d a1, ft1, rdn
# no-rv64-error@-1 {{operand must be 'rtz' floating-point rounding mode}}
# no-rv32-error@-2 {{operand must be 'rtz' floating-point rounding mode}}

fcvtmod.w.d a1, ft1, rup
# no-rv64-error@-1 {{operand must be 'rtz' floating-point rounding mode}}
# no-rv32-error@-2 {{operand must be 'rtz' floating-point rounding mode}}

# Invalid floating-point immediate
fli.s ft1, 5.250000e-01
# no-rv64-error@-1 {{operand must be a valid floating-point constant}}
# no-rv32-error@-2 {{operand must be a valid floating-point constant}}

fli.d ft1, 3.560000e+02
# no-rv64-error@-1 {{operand must be a valid floating-point constant}}
# no-rv32-error@-2 {{operand must be a valid floating-point constant}}

fli.h ft1, 1.600000e+00
# no-rv64-error@-1 {{operand must be a valid floating-point constant}}
# no-rv32-error@-2 {{operand must be a valid floating-point constant}}

fli.q ft1, 2.250000e+00
# no-rv64-error@-1 {{operand must be a valid floating-point constant}}
# no-rv32-error@-2 {{operand must be a valid floating-point constant}}

fli.s ft1, -min
# no-rv64-error@-1 {{invalid floating point immediate}}
# no-rv32-error@-2 {{invalid floating point immediate}}

fli.s ft1, -inf
# no-rv64-error@-1 {{invalid floating point immediate}}
# no-rv32-error@-2 {{invalid floating point immediate}}

fli.s ft1, -nan
# no-rv64-error@-1 {{invalid floating point immediate}}
# no-rv32-error@-2 {{invalid floating point immediate}}

# Don't accept decimal minimum.
fli.s ft1, 1.1754943508222875079687365372222456778186655567720875215087517062784172594547271728515625e-38
# no-rv64-error@-1 {{operand must be a valid floating-point constant}}
# no-rv32-error@-2 {{operand must be a valid floating-point constant}}

# Don't accept decimal minimum.
fli.d ft1, 2.225073858507201383090232717332404064219215980462331830553327416887204434813918195854283159012511020564067339731035811005152434161553460108856012385377718821130777993532002330479610147442583636071921565046942503734208375250806650616658158948720491179968591639648500635908770118304874799780887753749949451580451605050915399856582470818645113537935804992115981085766051992433352114352390148795699609591288891602992641511063466313393663477586513029371762047325631781485664350872122828637642044846811407613911477062801689853244110024161447421618567166150540154285084716752901903161322778896729707373123334086988983175067838846926092773977972858659654941091369095406136467568702398678315290680984617210924625396728515625e-308
# no-rv64-error@-1 {{operand must be a valid floating-point constant}}
# no-rv32-error@-2 {{operand must be a valid floating-point constant}}

# Don't accept decimal minimum.
fli.h ft1, 6.103516e-05
# no-rv64-error@-1 {{operand must be a valid floating-point constant}}
# no-rv32-error@-2 {{operand must be a valid floating-point constant}}

# Don't accept single precision minimum for double.
fli.d ft1, 1.1754943508222875079687365372222456778186655567720875215087517062784172594547271728515625e-38
# no-rv64-error@-1 {{operand must be a valid floating-point constant}}
# no-rv32-error@-2 {{operand must be a valid floating-point constant}}

# Don't accept single precision minimum for half.
fli.h ft1, 1.1754943508222875079687365372222456778186655567720875215087517062784172594547271728515625e-38
# no-rv64-error@-1 {{operand must be a valid floating-point constant}}
# no-rv32-error@-2 {{operand must be a valid floating-point constant}}

# Don't accept single precision minimum for quad.
fli.q ft1, 1.1754943508222875079687365372222456778186655567720875215087517062784172594547271728515625e-38
# no-rv64-error@-1 {{operand must be a valid floating-point constant}}
# no-rv32-error@-2 {{operand must be a valid floating-point constant}}

# Don't accept integers.
fli.s ft1, 1
# no-rv32-error@-1 {{invalid floating point immediate}}
# no-rv64-error@-2 {{invalid floating point immediate}}
