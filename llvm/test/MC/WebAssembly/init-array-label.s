# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj < %s | obj2yaml | FileCheck %s

init1:
	.functype	init1 () -> ()
	end_function

init2:
	.functype	init2 () -> ()
	end_function

	.section	.init_array.42,"",@
	.p2align	2, 0x0
	.int32	init1

	.section	.init_array,"",@
	.globl	p_init1
	.p2align	2, 0x0
p_init1:
	.int32	init1
	.size	p_init1, 4

	.section	.init_array,"",@
	.globl	p_init2
	.p2align	2, 0x0
p_init2:
	.int32	init1
	.int32	init2
	.size	p_init2, 8

# CHECK:        - Type:            FUNCTION
# CHECK-NEXT:     FunctionTypes:   [ 0, 0 ]
# CHECK-NEXT:   - Type:            DATACOUNT
# CHECK-NEXT:     Count:           1
# CHECK-NEXT:   - Type:            CODE
# CHECK-NEXT:     Functions:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Locals:          []
# CHECK-NEXT:         Body:            0B
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Locals:          []
# CHECK-NEXT:         Body:            0B
# CHECK-NEXT:   - Type:            DATA
# CHECK-NEXT:     Segments:
# CHECK-NEXT:       - SectionOffset:   6
# CHECK-NEXT:         InitFlags:       0
# CHECK-NEXT:         Offset:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           0
# CHECK-NEXT:         Content:         '000000000000000000000000'
# CHECK-NEXT:   - Type:            CUSTOM
# CHECK-NEXT:     Name:            linking
# CHECK-NEXT:     Version:         2
# CHECK-NEXT:     SymbolTable:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Name:            init1
# CHECK-NEXT:         Flags:           [ BINDING_LOCAL ]
# CHECK-NEXT:         Function:        0
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Name:            init2
# CHECK-NEXT:         Flags:           [ BINDING_LOCAL ]
# CHECK-NEXT:         Function:        1
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Kind:            DATA
# CHECK-NEXT:         Name:            p_init1
# CHECK-NEXT:         Flags:           [  ]
# CHECK-NEXT:         Segment:         0
# CHECK-NEXT:         Size:            4
# CHECK-NEXT:       - Index:           3
# CHECK-NEXT:         Kind:            DATA
# CHECK-NEXT:         Name:            p_init2
# CHECK-NEXT:         Flags:           [  ]
# CHECK-NEXT:         Segment:         0
# CHECK-NEXT:         Offset:          4
# CHECK-NEXT:         Size:            8
# CHECK-NEXT:     SegmentInfo:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Name:            .init_array
# CHECK-NEXT:         Alignment:       2
# CHECK-NEXT:         Flags:           [  ]
# CHECK-NEXT:     InitFunctions:
# CHECK-NEXT:       - Priority:        42
# CHECK-NEXT:         Symbol:          0
# CHECK-NEXT:       - Priority:        65535
# CHECK-NEXT:         Symbol:          0
# CHECK-NEXT:       - Priority:        65535
# CHECK-NEXT:         Symbol:          0
# CHECK-NEXT:       - Priority:        65535
# CHECK-NEXT:         Symbol:          1
# CHECK-NEXT: ...
