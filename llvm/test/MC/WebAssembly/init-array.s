# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj < %s | obj2yaml | FileCheck %s

init1:
	.functype	init1 () -> ()
	end_function

init2:
	.functype	init2 () -> ()
	end_function

	.section	.init_array,"",@
	.p2align	2, 0
	.int32	init1

	.section	.init_array,"",@
	.p2align	2
	.int32	init2

# CHECK:        - Type:            FUNCTION
# CHECK-NEXT:     FunctionTypes:   [ 0, 0 ]
# CHECK-NEXT:   - Type:            CODE
# CHECK-NEXT:     Functions:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Locals:          []
# CHECK-NEXT:         Body:            0B
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Locals:          []
# CHECK-NEXT:         Body:            0B
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
# CHECK-NEXT:     InitFunctions:
# CHECK-NEXT:       - Priority:        65535
# CHECK-NEXT:         Symbol:          0
# CHECK-NEXT:       - Priority:        65535
# CHECK-NEXT:         Symbol:          1
# CHECK-NEXT: ...
#
