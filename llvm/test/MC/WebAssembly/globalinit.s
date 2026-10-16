# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj %s -o %t.o
# RUN: obj2yaml %t.o | FileCheck %s --check-prefix=OBJ
# RUN: llvm-mc -triple=wasm32-unknown-unknown %s | llvm-mc -triple=wasm32-unknown-unknown -filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=OBJ
# RUN: llvm-mc -triple=wasm32-unknown-unknown %s | FileCheck %s --check-prefix=ASM

	.globaltype	i32g, i32
	.globalinit	i32g, i32.const 33554435
	.globl	i32g
i32g:

	.globaltype	i64g, i64
	.globalinit	i64g, i64.const -2
	.globl	i64g
i64g:

	.globaltype	f32g, f32
	.globalinit	f32g, f32.const 0x1p1
	.globl	f32g
f32g:

	.globaltype	f64g, f64
	.globalinit	f64g, f64.const -0x0p0
	.globl	f64g
f64g:

	.globaltype	nang, f32
	.globalinit	nang, f32.const nan
	.globl	nang
nang:

	.globaltype	constg, i32, immutable
	.globalinit	constg, i32.const 42
	.globl	constg
constg:

# Either spelling of a value that fits the type's width is accepted.

	.globaltype	u32max, i32
	.globalinit	u32max, i32.const 4294967295
	.globl	u32max
u32max:

	.globaltype	i32min, i32
	.globalinit	i32min, i32.const -2147483648
	.globl	i32min
i32min:

	.globaltype	i32minu, i32
	.globalinit	i32minu, i32.const 2147483648
	.globl	i32minu
i32minu:

	.globaltype	u64max, i64
	.globalinit	u64max, i64.const 18446744073709551615
	.globl	u64max
u64max:

# Repeating a declaration is idempotent, both for the type and for the value.

	.globaltype	retyped, i32
	.globalinit	retyped, i32.const 7
	.globaltype	retyped, i32
	.globl	retyped
retyped:

	.globaltype	twice, i32
	.globalinit	twice, i32.const 9
	.globalinit	twice, i32.const 9
	.globl	twice
twice:

# A repeat is compared on the union member its opcode selects, so a wider value
# written for another symbol in between cannot make it look like a conflict.

	.globaltype	wide, f64
	.globalinit	wide, f64.const 0x1.23456789abcdep100
	.globl	wide
wide:

	.globaltype	narrow, i32
	.globalinit	narrow, i32.const 3
	.globalinit	wide, f64.const 0x1.23456789abcdep100
	.globalinit	narrow, i32.const 3
	.globl	narrow
narrow:

	.globaltype	inf32, f32
	.globalinit	inf32, f32.const infinity
	.globl	inf32
inf32:

	.globaltype	dec32, f32
	.globalinit	dec32, f32.const 0.5
	.globl	dec32
dec32:

# A negative NaN keeps its sign.

	.globaltype	nneg64, f64
	.globalinit	nneg64, f64.const -nan
	.globl	nneg64
nneg64:

	.globaltype	nneg32, f32
	.globalinit	nneg32, f32.const -nan
	.globl	nneg32
nneg32:

# A value equal to the type's default is printed back.

	.globaltype	zerog, i32
	.globalinit	zerog, i32.const 0
	.globl	zerog
zerog:

# ASM:      .globaltype i32g, i32
# ASM-NEXT: .globalinit i32g, i32.const 33554435
# ASM:      .globaltype i64g, i64
# ASM-NEXT: .globalinit i64g, i64.const -2
# ASM:      .globaltype f32g, f32
# ASM-NEXT: .globalinit f32g, f32.const 0x1p1
# ASM:      .globaltype f64g, f64
# ASM-NEXT: .globalinit f64g, f64.const -0x0p0
# ASM:      .globaltype nang, f32
# ASM-NEXT: .globalinit nang, f32.const nan
# ASM:      .globaltype constg, i32, immutable
# ASM-NEXT: .globalinit constg, i32.const 42
# ASM:      .globaltype u32max, i32
# ASM-NEXT: .globalinit u32max, i32.const -1
# ASM:      .globaltype i32min, i32
# ASM-NEXT: .globalinit i32min, i32.const -2147483648
# ASM:      .globaltype i32minu, i32
# ASM-NEXT: .globalinit i32minu, i32.const -2147483648
# ASM:      .globaltype u64max, i64
# ASM-NEXT: .globalinit u64max, i64.const -1
# ASM:      .globaltype retyped, i32
# ASM-NEXT: .globalinit retyped, i32.const 7
# ASM-NEXT: .globaltype retyped, i32
# ASM:      .globaltype twice, i32
# ASM-NEXT: .globalinit twice, i32.const 9
# ASM-NEXT: .globalinit twice, i32.const 9
# ASM:      .globaltype wide, f64
# ASM-NEXT: .globalinit wide, f64.const 0x1.23456789abcdep100
# ASM:      .globaltype narrow, i32
# ASM-NEXT: .globalinit narrow, i32.const 3
# ASM-NEXT: .globalinit wide, f64.const 0x1.23456789abcdep100
# ASM-NEXT: .globalinit narrow, i32.const 3
# ASM:      .globaltype inf32, f32
# ASM-NEXT: .globalinit inf32, f32.const infinity
# ASM:      .globaltype dec32, f32
# ASM-NEXT: .globalinit dec32, f32.const 0x1p-1
# ASM:      .globaltype nneg64, f64
# ASM-NEXT: .globalinit nneg64, f64.const -nan
# ASM:      .globaltype nneg32, f32
# ASM-NEXT: .globalinit nneg32, f32.const -nan
# ASM:      .globaltype zerog, i32
# ASM-NEXT: .globalinit zerog, i32.const 0

# OBJ:      - Type:            GLOBAL
# OBJ-NEXT:   Globals:
# OBJ-NEXT:     - Index:           0
# OBJ-NEXT:       Type:            I32
# OBJ-NEXT:       Mutable:         true
# OBJ-NEXT:       InitExpr:
# OBJ-NEXT:         Opcode:          I32_CONST
# OBJ-NEXT:         Value:           33554435
# OBJ-NEXT:     - Index:           1
# OBJ-NEXT:       Type:            I64
# OBJ-NEXT:       Mutable:         true
# OBJ-NEXT:       InitExpr:
# OBJ-NEXT:         Opcode:          I64_CONST
# OBJ-NEXT:         Value:           -2
# OBJ-NEXT:     - Index:           2
# OBJ-NEXT:       Type:            F32
# OBJ-NEXT:       Mutable:         true
# OBJ-NEXT:       InitExpr:
# OBJ-NEXT:         Opcode:          F32_CONST
# OBJ-NEXT:         Value:           1073741824
# OBJ-NEXT:     - Index:           3
# OBJ-NEXT:       Type:            F64
# OBJ-NEXT:       Mutable:         true
# OBJ-NEXT:       InitExpr:
# OBJ-NEXT:         Opcode:          F64_CONST
# OBJ-NEXT:         Value:           9223372036854775808
# OBJ-NEXT:     - Index:           4
# OBJ-NEXT:       Type:            F32
# OBJ-NEXT:       Mutable:         true
# OBJ-NEXT:       InitExpr:
# OBJ-NEXT:         Opcode:          F32_CONST
# OBJ-NEXT:         Value:           2143289344
# OBJ-NEXT:     - Index:           5
# OBJ-NEXT:       Type:            I32
# OBJ-NEXT:       Mutable:         false
# OBJ-NEXT:       InitExpr:
# OBJ-NEXT:         Opcode:          I32_CONST
# OBJ-NEXT:         Value:           42
# OBJ-NEXT:     - Index:           6
# OBJ-NEXT:       Type:            I32
# OBJ-NEXT:       Mutable:         true
# OBJ-NEXT:       InitExpr:
# OBJ-NEXT:         Opcode:          I32_CONST
# OBJ-NEXT:         Value:           -1
# OBJ-NEXT:     - Index:           7
# OBJ-NEXT:       Type:            I32
# OBJ-NEXT:       Mutable:         true
# OBJ-NEXT:       InitExpr:
# OBJ-NEXT:         Opcode:          I32_CONST
# OBJ-NEXT:         Value:           -2147483648
# OBJ-NEXT:     - Index:           8
# OBJ-NEXT:       Type:            I32
# OBJ-NEXT:       Mutable:         true
# OBJ-NEXT:       InitExpr:
# OBJ-NEXT:         Opcode:          I32_CONST
# OBJ-NEXT:         Value:           -2147483648
# OBJ-NEXT:     - Index:           9
# OBJ-NEXT:       Type:            I64
# OBJ-NEXT:       Mutable:         true
# OBJ-NEXT:       InitExpr:
# OBJ-NEXT:         Opcode:          I64_CONST
# OBJ-NEXT:         Value:           -1
# OBJ-NEXT:     - Index:           10
# OBJ-NEXT:       Type:            I32
# OBJ-NEXT:       Mutable:         true
# OBJ-NEXT:       InitExpr:
# OBJ-NEXT:         Opcode:          I32_CONST
# OBJ-NEXT:         Value:           7
# OBJ-NEXT:     - Index:           11
# OBJ-NEXT:       Type:            I32
# OBJ-NEXT:       Mutable:         true
# OBJ-NEXT:       InitExpr:
# OBJ-NEXT:         Opcode:          I32_CONST
# OBJ-NEXT:         Value:           9
# OBJ-NEXT:     - Index:           12
# OBJ-NEXT:       Type:            F64
# OBJ-NEXT:       Mutable:         true
# OBJ-NEXT:       InitExpr:
# OBJ-NEXT:         Opcode:          F64_CONST
# OBJ-NEXT:         Value:           5058162877485726942
# OBJ-NEXT:     - Index:           13
# OBJ-NEXT:       Type:            I32
# OBJ-NEXT:       Mutable:         true
# OBJ-NEXT:       InitExpr:
# OBJ-NEXT:         Opcode:          I32_CONST
# OBJ-NEXT:         Value:           3
# OBJ-NEXT:     - Index:           14
# OBJ-NEXT:       Type:            F32
# OBJ-NEXT:       Mutable:         true
# OBJ-NEXT:       InitExpr:
# OBJ-NEXT:         Opcode:          F32_CONST
# OBJ-NEXT:         Value:           2139095040
# OBJ-NEXT:     - Index:           15
# OBJ-NEXT:       Type:            F32
# OBJ-NEXT:       Mutable:         true
# OBJ-NEXT:       InitExpr:
# OBJ-NEXT:         Opcode:          F32_CONST
# OBJ-NEXT:         Value:           1056964608
# OBJ-NEXT:     - Index:           16
# OBJ-NEXT:       Type:            F64
# OBJ-NEXT:       Mutable:         true
# OBJ-NEXT:       InitExpr:
# OBJ-NEXT:         Opcode:          F64_CONST
# OBJ-NEXT:         Value:           18444492273895866368
# OBJ-NEXT:     - Index:           17
# OBJ-NEXT:       Type:            F32
# OBJ-NEXT:       Mutable:         true
# OBJ-NEXT:       InitExpr:
# OBJ-NEXT:         Opcode:          F32_CONST
# OBJ-NEXT:         Value:           4290772992
# OBJ-NEXT:     - Index:           18
# OBJ-NEXT:       Type:            I32
# OBJ-NEXT:       Mutable:         true
# OBJ-NEXT:       InitExpr:
# OBJ-NEXT:         Opcode:          I32_CONST
# OBJ-NEXT:         Value:           0
