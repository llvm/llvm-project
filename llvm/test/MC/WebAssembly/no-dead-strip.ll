; RUN: llc < %s --mtriple=wasm32-unknown-unknown -filetype=obj -wasm-keep-registers -o - | obj2yaml - | FileCheck %s

@llvm.used = appending global [5 x ptr] [
  ptr @foo, ptr @gv0, ptr @gv1, ptr @gv2, ptr @gv3
], section "llvm.metadata"

define i32 @foo() {
entry:
    ret i32 0
}

; externally visible GV has NO_STRIP/RETAIN in both symtab entry and segment info
@gv0 = global i32 42
; internal GV has NO_STRIP/RETAIN in both symtab entry and segment info
@gv1 = internal global i32 41
; private GV has RETAIN in segment info only (no symtab entry)
@gv2 = private global i32 40
; explicit section names
@gv3 = global i32 39, section "ddd.hello"
@gv4.not.used = global i64 38, section "ddd.hello"

; CHECK:         SymbolTable:
; CHECK-NEXT:      - Index:           0
; CHECK-NEXT:        Kind:            FUNCTION
; CHECK-NEXT:        Name:            foo
; CHECK-NEXT:        Flags:           [ NO_STRIP ]
; CHECK-NEXT:        Function:        0
; CHECK-NEXT:      - Index:           1
; CHECK-NEXT:        Kind:            DATA
; CHECK-NEXT:        Name:            gv0
; CHECK-NEXT:        Flags:           [ NO_STRIP ]
; CHECK-NEXT:        Segment:         0
; CHECK-NEXT:        Size:            4
; CHECK-NEXT:      - Index:           2
; CHECK-NEXT:        Kind:            DATA
; CHECK-NEXT:        Name:            gv1
; CHECK-NEXT:        Flags:           [ BINDING_LOCAL, NO_STRIP ]
; CHECK-NEXT:        Segment:         1
; CHECK-NEXT:        Size:            4
; CHECK-NEXT:      - Index:           3
; CHECK-NEXT:        Kind:            DATA
; CHECK-NEXT:        Name:            gv3
; CHECK-NEXT:        Flags:           [ NO_STRIP ]
; CHECK-NEXT:        Segment:         3
; CHECK-NEXT:        Size:            4
; CHECK-NEXT:      - Index:           4
; CHECK-NEXT:        Kind:            DATA
; CHECK-NEXT:        Name:            gv4.not.used
; CHECK-NEXT:        Flags:           [  ]
; CHECK-NEXT:        Segment:         3
; CHECK-NEXT:        Offset:          8
; CHECK-NEXT:        Size:            8
; CHECK-NEXT:    SegmentInfo:
; CHECK-NEXT:      - Index:           0
; CHECK-NEXT:        Name:            .data.gv0
; CHECK-NEXT:        Alignment:       2
; CHECK-NEXT:        Flags:           [ RETAIN ]
; CHECK-NEXT:      - Index:           1
; CHECK-NEXT:        Name:            .data.gv1
; CHECK-NEXT:        Alignment:       2
; CHECK-NEXT:        Flags:           [ RETAIN ]
; CHECK-NEXT:      - Index:           2
; CHECK-NEXT:        Name:            .data..Lgv2
; CHECK-NEXT:        Alignment:       2
; CHECK-NEXT:        Flags:           [ RETAIN ]
; CHECK-NEXT:      - Index:           3
; CHECK-NEXT:        Name:            ddd.hello
; CHECK-NEXT:        Alignment:       3
; CHECK-NEXT:        Flags:           [ RETAIN ]
