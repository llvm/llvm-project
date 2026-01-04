// RUN: llvm-mc -triple i386-pc-win32 -filetype=obj %s | llvm-readobj -S - | FileCheck %s
// RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj %s | llvm-readobj -S - | FileCheck %s
// RUN: not llvm-mc -triple x86_64-pc-win32 -filetype=obj --defsym ERR=1 %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

.section .foo$bar; .long 1
.section .foo@bar; .long 1
.section ABCDEFGHIJKLMNOPQRSTUVWXYZ; .long 1
.section abcdefghijklmnopqrstuvwxyz; .long 1
.section _0123456789; .long 1

// CHECK: Sections [
// CHECK:   Section {
// CHECK:     Name: .foo$bar
// CHECK:   }
// CHECK:   Section {
// CHECK:     Name: .foo@bar
// CHECK:   }
// CHECK:   Section {
// CHECK:     Name: ABCDEFGHIJKLMNOPQRSTUVWXYZ
// CHECK:   }
// CHECK:   Section {
// CHECK:     Name: abcdefghijklmnopqrstuvwxyz
// CHECK:   }
// CHECK:   Section {
// CHECK:     Name: _0123456789
// CHECK:   }

// Test that the defaults are used
.section s      ; .long 1
.section s_, "" ; .long 1
.section s_a,"a"; .long 1
.section s_b,"b"; .long 0
.section s_d,"d"; .long 1
.section s_D,"D"; .long 1
.section s_n,"n"; .long 1
.section s_r,"r"; .long 1
.section s_s,"s"; .long 1
.section s_w,"w"; .long 1
.section s_x,"x"; .long 1
.section s_y,"y"; .long 1
.section s_i,"i"; .long 1

// CHECK:        Section {
// CHECK:          Name: s
// CHECK:          Characteristics [
// CHECK-NEXT:       IMAGE_SCN_ALIGN_1BYTES
// CHECK-NEXT:       IMAGE_SCN_CNT_INITIALIZED_DATA
// CHECK-NEXT:       IMAGE_SCN_MEM_READ
// CHECK-NEXT:       IMAGE_SCN_MEM_WRITE
// CHECK-NEXT:     ]
// CHECK:        }
// CHECK:        Section {
// CHECK:          Name: s_
// CHECK:          Characteristics [
// CHECK-NEXT:       IMAGE_SCN_ALIGN_1BYTES
// CHECK-NEXT:       IMAGE_SCN_CNT_INITIALIZED_DATA
// CHECK-NEXT:       IMAGE_SCN_MEM_READ
// CHECK-NEXT:       IMAGE_SCN_MEM_WRITE
// CHECK-NEXT:     ]
// CHECK:        }
// CHECK:        Section {
// CHECK:          Name: s_a
// CHECK:          Characteristics [
// CHECK-NEXT:       IMAGE_SCN_ALIGN_1BYTES
// CHECK-NEXT:       IMAGE_SCN_CNT_INITIALIZED_DATA
// CHECK-NEXT:       IMAGE_SCN_MEM_READ
// CHECK-NEXT:       IMAGE_SCN_MEM_WRITE
// CHECK-NEXT:     ]
// CHECK:        }
// CHECK:        Section {
// CHECK:          Name: s_b
// CHECK:          Characteristics [
// CHECK-NEXT:       IMAGE_SCN_ALIGN_1BYTES
// CHECK-NEXT:       IMAGE_SCN_CNT_UNINITIALIZED_DATA
// CHECK-NEXT:       IMAGE_SCN_MEM_READ
// CHECK-NEXT:       IMAGE_SCN_MEM_WRITE
// CHECK-NEXT:     ]
// CHECK:        }
// CHECK:        Section {
// CHECK:          Name: s_d
// CHECK:          Characteristics [
// CHECK-NEXT:       IMAGE_SCN_ALIGN_1BYTES
// CHECK-NEXT:       IMAGE_SCN_CNT_INITIALIZED_DATA
// CHECK-NEXT:       IMAGE_SCN_MEM_READ
// CHECK-NEXT:       IMAGE_SCN_MEM_WRITE
// CHECK-NEXT:     ]
// CHECK:        }
// CHECK:        Section {
// CHECK:          Name: s_D
// CHECK:          Characteristics [
// CHECK-NEXT:       IMAGE_SCN_ALIGN_1BYTES
// CHECK-NEXT:       IMAGE_SCN_MEM_DISCARDABLE
// CHECK-NEXT:       IMAGE_SCN_MEM_READ
// CHECK-NEXT:       IMAGE_SCN_MEM_WRITE
// CHECK-NEXT:     ]
// CHECK:        }
// CHECK:        Section {
// CHECK:          Name: s_n
// CHECK:          Characteristics [
// CHECK-NEXT:       IMAGE_SCN_ALIGN_1BYTES
// CHECK-NEXT:       IMAGE_SCN_LNK_REMOVE
// CHECK-NEXT:       IMAGE_SCN_MEM_READ
// CHECK-NEXT:       IMAGE_SCN_MEM_WRITE
// CHECK-NEXT:     ]
// CHECK:        }
// CHECK:        Section {
// CHECK:          Name: s_r
// CHECK:          Characteristics [
// CHECK-NEXT:       IMAGE_SCN_ALIGN_1BYTES
// CHECK-NEXT:       IMAGE_SCN_CNT_INITIALIZED_DATA
// CHECK-NEXT:       IMAGE_SCN_MEM_READ
// CHECK-NEXT:     ]
// CHECK:        }
// CHECK:        Section {
// CHECK:          Name: s_s
// CHECK:          Characteristics [
// CHECK-NEXT:       IMAGE_SCN_ALIGN_1BYTES
// CHECK-NEXT:       IMAGE_SCN_CNT_INITIALIZED_DATA
// CHECK-NEXT:       IMAGE_SCN_MEM_READ
// CHECK-NEXT:       IMAGE_SCN_MEM_SHARED
// CHECK-NEXT:       IMAGE_SCN_MEM_WRITE
// CHECK-NEXT:     ]
// CHECK:        }
// CHECK:        Section {
// CHECK:          Name: s_w
// CHECK:          Characteristics [
// CHECK-NEXT:       IMAGE_SCN_ALIGN_1BYTES
// CHECK-NEXT:       IMAGE_SCN_CNT_INITIALIZED_DATA
// CHECK-NEXT:       IMAGE_SCN_MEM_READ
// CHECK-NEXT:       IMAGE_SCN_MEM_WRITE
// CHECK-NEXT:     ]
// CHECK:        }
// CHECK:        Section {
// CHECK:          Name: s_x
// CHECK:          Characteristics [
// CHECK-NEXT:       IMAGE_SCN_ALIGN_1BYTES
// CHECK-NEXT:       IMAGE_SCN_CNT_CODE
// CHECK-NEXT:       IMAGE_SCN_MEM_EXECUTE
// CHECK-NEXT:       IMAGE_SCN_MEM_READ
// CHECK-NEXT:     ]
// CHECK:        }
// CHECK:        Section {
// CHECK:          Name: s_y
// CHECK:          Characteristics [
// CHECK-NEXT:       IMAGE_SCN_ALIGN_1BYTES
// CHECK-NEXT:     ]
// CHECK:        }
// CHECK:        Section {
// CHECK:          Name: s_i
// CHECK:          Characteristics [
// CHECK-NEXT:       IMAGE_SCN_ALIGN_1BYTES
// CHECK-NEXT:       IMAGE_SCN_LNK_INFO
// CHECK-NEXT:       IMAGE_SCN_MEM_READ
// CHECK-NEXT:       IMAGE_SCN_MEM_WRITE
// CHECK-NEXT:     ]
// CHECK:        }

// w makes read-only to readable
.section s_rw,"rw"; .long 1
// CHECK:        Section {
// CHECK:          Name: s_rw
// CHECK:          Characteristics [
// CHECK-NEXT:       IMAGE_SCN_ALIGN_1BYTES
// CHECK-NEXT:       IMAGE_SCN_CNT_INITIALIZED_DATA
// CHECK-NEXT:       IMAGE_SCN_MEM_READ
// CHECK-NEXT:       IMAGE_SCN_MEM_WRITE
// CHECK-NEXT:     ]
// CHECK:        }

// r cancels w
.section s_wr,"wr"; .long 1
// CHECK:        Section {
// CHECK:          Name: s_wr
// CHECK:          Characteristics [
// CHECK-NEXT:       IMAGE_SCN_ALIGN_1BYTES
// CHECK-NEXT:       IMAGE_SCN_CNT_INITIALIZED_DATA
// CHECK-NEXT:       IMAGE_SCN_MEM_READ
// CHECK-NEXT:     ]
// CHECK:        }

// y cancels both
.section s_rwy,"rwy"; .long 1
// CHECK:        Section {
// CHECK:          Name: s_rwy
// CHECK:          Characteristics [
// CHECK-NEXT:       IMAGE_SCN_ALIGN_1BYTES
// CHECK-NEXT:       IMAGE_SCN_CNT_INITIALIZED_DATA
// CHECK-NEXT:     ]
// CHECK:        }

// Sections starting with ".debug" are implicitly discardable. This is
// compatible with gas.
.section .debug_asdf,"dr"; .long 1
// CHECK:        Section {
// CHECK:          Name: .debug_asdf
// CHECK:          Characteristics [
// CHECK-NEXT:       IMAGE_SCN_ALIGN_1BYTES
// CHECK-NEXT:       IMAGE_SCN_CNT_INITIALIZED_DATA
// CHECK-NEXT:       IMAGE_SCN_MEM_DISCARDABLE
// CHECK-NEXT:       IMAGE_SCN_MEM_READ
// CHECK-NEXT:     ]
// CHECK:        }

/// The section name can be quoted.
.section "@#$-{","n"
// CHECK:        Section {
// CHECK-NEXT:     Number:
// CHECK-NEXT:     Name: @#$-{

// CHECK-NOT:    Section {
// CHECK:      ]

.section data1; .quad 0

.pushsection data2; .quad 0
.popsection

// Back to section data1
.quad 2

// CHECK:       Section {
// CHECK-NEXT:    Number:
// CHECK-NEXT:    Name: data1
// CHECK:         RawDataSize: 16

// CHECK:       Section {
// CHECK-NEXT:    Number:
// CHECK-NEXT:    Name: data2
// CHECK:         RawDataSize: 8

.section .data3,"dw"; .quad 1

.pushsection .data4,"dw"; .quad 1
.popsection

.pushsection .data5,"dr"; .quad 1
.popsection

// in section .data3
.quad 4

// Notice the different section flags here.
// This shouldn't overwrite the intial section flags.
.pushsection .data4,"dr"; .quad 1
.popsection

// CHECK:       Section {
// CHECK-NEXT:    Number:
// CHECK-NEXT:    Name: .data3
// CHECK:         RawDataSize: 16
// CHECK:         Characteristics [
// CHECK-NEXT:      IMAGE_SCN_ALIGN_1BYTES
// CHECK-NEXT:      IMAGE_SCN_CNT_INITIALIZED_DATA
// CHECK-NEXT:      IMAGE_SCN_MEM_READ
// CHECK-NEXT:      IMAGE_SCN_MEM_WRITE
// CHECK-NEXT:    ]

// CHECK:       Section {
// CHECK-NEXT:    Number:
// CHECK-NEXT:    Name: .data4
// CHECK:         RawDataSize: 16
// CHECK:         Characteristics [
// CHECK-NEXT:      IMAGE_SCN_ALIGN_1BYTES
// CHECK-NEXT:      IMAGE_SCN_CNT_INITIALIZED_DATA
// CHECK-NEXT:      IMAGE_SCN_MEM_READ
// CHECK-NEXT:      IMAGE_SCN_MEM_WRITE
// CHECK-NEXT:    ]

// CHECK:       Section {
// CHECK-NEXT:    Number:
// CHECK-NEXT:    Name: .data5
// CHECK:         RawDataSize: 8
// CHECK:         Characteristics [
// CHECK-NEXT:      IMAGE_SCN_ALIGN_1BYTES
// CHECK-NEXT:      IMAGE_SCN_CNT_INITIALIZED_DATA
// CHECK-NEXT:      IMAGE_SCN_MEM_READ
// CHECK-NEXT:    ]

.ifdef ERR
// ERR: :[[#@LINE+1]]:12: error: .popsection without corresponding .pushsection
.popsection
.endif
