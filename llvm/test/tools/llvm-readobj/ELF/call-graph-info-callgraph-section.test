## Tests --call-graph-info prints information from call graph section.

# REQUIRES: x86-registered-target

# RUN: llvm-mc %s -filetype=obj -triple=x86_64-pc-linux -o %t
# RUN: llvm-readelf --call-graph-info %t 2>&1 | FileCheck %s --allow-empty -DFILE=%t
# RUN: llvm-readelf --elf-output-style=LLVM --call-graph-info %t 2>&1 | FileCheck %s -DFILE=%t --check-prefix=LLVM
# RUN: llvm-readelf --elf-output-style=JSON --pretty-print --call-graph-info %t 2>&1 | FileCheck %s -DFILE=%t --check-prefix=JSON


## We do not support GNU format console output for --call-graph-info as it is an LLVM only info.
# CHECK-NOT: .

# LLVM: warning: '[[FILE]]': .llvm.callgraph section has unknown type id for 2 indirect targets.
# LLVM-NEXT: CallGraph [
# LLVM-NEXT:   Function {
# LLVM-NEXT:     Offset: 0x2
# LLVM-NEXT:     Version: 0
# LLVM-NEXT:     IsIndirectTarget: Yes
# LLVM-NEXT:     TypeId: 0x0
# LLVM-NEXT:     NumDirectCallees: 1
# LLVM-NEXT:     DirectCallees [
# LLVM-NEXT:       {
# LLVM-NEXT:         Offset: 0x13
# LLVM-NEXT:       }
# LLVM-NEXT:     ]
# LLVM-NEXT:     NumIndirectTargetTypeIDs: 0
# LLVM-NEXT:     IndirectTypeIDs: []
# LLVM-NEXT:   }
# LLVM-NEXT:   Function {
# LLVM-NEXT:     Offset: 0x1D
# LLVM-NEXT:     Version: 0
# LLVM-NEXT:     IsIndirectTarget: Yes
# LLVM-NEXT:     TypeId: 0x0
# LLVM-NEXT:     NumDirectCallees: 0
# LLVM-NEXT:     DirectCallees [
# LLVM-NEXT:     ]
# LLVM-NEXT:     NumIndirectTargetTypeIDs: 1
# LLVM-NEXT:     IndirectTypeIDs: [0x10]
# LLVM-NEXT:   }
# LLVM-NEXT:   Function {
# LLVM-NEXT:     Offset: 0x38
# LLVM-NEXT:     Version: 0
# LLVM-NEXT:     IsIndirectTarget: Yes
# LLVM-NEXT:     TypeId: 0x20
# LLVM-NEXT:     NumDirectCallees: 0
# LLVM-NEXT:     DirectCallees [
# LLVM-NEXT:     ]
# LLVM-NEXT:     NumIndirectTargetTypeIDs: 0
# LLVM-NEXT:     IndirectTypeIDs: []
# LLVM-NEXT:   }
# LLVM-NEXT: ]

# JSON: warning: '[[FILE]]': .llvm.callgraph section has unknown type id for 2 indirect targets.
# JSON:     "CallGraph": [
# JSON-NEXT:      {
# JSON-NEXT:        "Function": {
# JSON-NEXT:          "Offset": 2,
# JSON-NEXT:          "Version": 0,
# JSON-NEXT:          "IsIndirectTarget": true,
# JSON-NEXT:          "TypeId": 0,
# JSON-NEXT:          "NumDirectCallees": 1,
# JSON-NEXT:          "DirectCallees": [
# JSON-NEXT:            {
# JSON-NEXT:              "Offset": 19
# JSON-NEXT:            }
# JSON-NEXT:          ],
# JSON-NEXT:          "NumIndirectTargetTypeIDs": 0,
# JSON-NEXT:          "IndirectTypeIDs": []
# JSON-NEXT:        }
# JSON-NEXT:      },
# JSON-NEXT:      {
# JSON-NEXT:        "Function": {
# JSON-NEXT:          "Offset": 29,
# JSON-NEXT:          "Version": 0,
# JSON-NEXT:          "IsIndirectTarget": true,
# JSON-NEXT:          "TypeId": 0,
# JSON-NEXT:          "NumDirectCallees": 0,
# JSON-NEXT:          "DirectCallees": [],
# JSON-NEXT:          "NumIndirectTargetTypeIDs": 1,
# JSON-NEXT:          "IndirectTypeIDs": [
# JSON-NEXT:            16
# JSON-NEXT:          ]
# JSON-NEXT:        }
# JSON-NEXT:      },
# JSON-NEXT:      {
# JSON-NEXT:        "Function": {
# JSON-NEXT:          "Offset": 56,
# JSON-NEXT:          "Version": 0,
# JSON-NEXT:          "IsIndirectTarget": true,
# JSON-NEXT:          "TypeId": 32,
# JSON-NEXT:          "NumDirectCallees": 0,
# JSON-NEXT:          "DirectCallees": [],
# JSON-NEXT:          "NumIndirectTargetTypeIDs": 0,
# JSON-NEXT:          "IndirectTypeIDs": []
# JSON-NEXT:        }
# JSON-NEXT:      }
# JSON-NEXT:    ]
# JSON-NEXT:  }
# JSON-NEXT:]

.text

.globl foo
.type foo,@function
foo:                  #< foo is at 0.
.Lfoo_begin:
 callq foo            #< direct call is at 5. target is foo (5).
 retq

.globl bar
.type bar,@function
bar:                  #< bar is at 6.
 callq	*-40(%rbp)    #< indirect call is at 9.
 retq

.globl baz
.type baz,@function
baz:                  #< baz is at 10 (a).
 retq

.globl qux
.type qux,@function
qux:                  #< qux is at 11 (b).
 retq

.section	.llvm.callgraph,"o",@llvm_call_graph,.text
.byte	0       #< Format version number.
.byte	3       #< Flag IsIndirectTarget true
.quad	0       #< foo()'s address.
.quad	0       #< TypeID: unknown.
.byte   1       #< Count of direct callees.
.quad   5       #< Direct callee foo's address>

.byte	0       #< Format version number.
.byte   5       #< Flag IsIndirectTarget true
.quad	6       #< bar()'s address.
.quad	0       #< TypeID: unknown.
.byte	1       #< Count of indirect target type IDs
.quad   16      #< Indirect call type id.


.byte	0       #< Format version number.
.byte   1       #< Flag IsIndirectTarget true
.quad	10      #< baz()'s address.
.quad   32      #< Indirect target type id.

# No call graph section entry for qux. 

.text
