// RUN: llvm-mc -triple i386-pc-win32 -filetype=obj %s | llvm-readobj -S - | FileCheck %s
// RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj %s | llvm-readobj -S - | FileCheck %s

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
.section s_b,"b"; .long 1
.section s_d,"d"; .long 1
.section s_D,"D"; .long 1
.section s_n,"n"; .long 1
.section s_r,"r"; .long 1
.section s_s,"s"; .long 1
.section s_w,"w"; .long 1
.section s_x,"x"; .long 1
.section s_y,"y"; .long 1

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
