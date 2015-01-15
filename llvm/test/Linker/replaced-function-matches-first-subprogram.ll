; RUN: llvm-link %s %S/Inputs/replaced-function-matches-first-subprogram.ll -S | FileCheck %s

; Generated from C++ source:
;
; // repro/t.h
; template <class T> struct Class {
;   int foo() { return 0; }
; };
; // repro/d1/t1.cpp
; #include "t.h"
; int foo() { return Class<int>().foo(); }
; // repro/d2/t2.cpp
; #include "t.h"
; template struct Class<int>;

%struct.Class = type { i8 }

define i32 @_Z3foov() {
entry:
  %tmp = alloca %struct.Class, align 1
  %call = call i32 @_ZN5ClassIiE3fooEv(%struct.Class* %tmp), !dbg !14
  ret i32 %call, !dbg !14
}

; CHECK: define weak_odr i32 @_ZN5ClassIiE3fooEv(%struct.Class* %this){{.*}}{
; CHECK-NOT: }
; CHECK: !dbg ![[LOC:[0-9]+]]
define linkonce_odr i32 @_ZN5ClassIiE3fooEv(%struct.Class* %this) align 2 {
entry:
  %this.addr = alloca %struct.Class*, align 8
  store %struct.Class* %this, %struct.Class** %this.addr, align 8
  %this1 = load %struct.Class** %this.addr
  ret i32 0, !dbg !15
}

; CHECK: !llvm.dbg.cu = !{![[CU1:[0-9]+]], ![[CU2:[0-9]+]]}
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11, !12}
!llvm.ident = !{!13}

; Extract out the list of subprograms from each compile unit.
; CHECK-DAG: ![[CU1]] = !{!"0x11{{[^"]+}}", {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, ![[SPs1:[0-9]+]],
; CHECK-DAG: ![[CU2]] = !{!"0x11{{[^"]+}}", {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, ![[SPs2:[0-9]+]],
!0 = !{!"0x11\004\00clang version 3.6.0 (trunk 224193) (llvm/trunk 224197)\000\00\000\00\002", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/Users/dexonsmith/data/llvm/staging/test/Linker/repro/d1/t1.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"t1.cpp", !"/Users/dexonsmith/data/llvm/staging/test/Linker/repro/d1"}
!2 = !{}

; Extract out each compile unit's single subprogram.  The replaced subprogram
; should be dropped by the first compile unit.
; CHECK-DAG: ![[SPs1]] = !{![[SP1:[0-9]+]]}
; CHECK-DAG: ![[SPs2]] = !{![[SP2:[0-9]+]]}
!3 = !{!4, !7}
!4 = !{!"0x2e\00foo\00foo\00\002\000\001\000\000\00256\000\002", !1, !5, !6, null, i32 ()* @_Z3foov, null, null, !2} ; [ DW_TAG_subprogram ] [line 2] [def] [foo]
!5 = !{!"0x29", !1}    ; [ DW_TAG_file_type ] [/Users/dexonsmith/data/llvm/staging/test/Linker/repro/d1/t1.cpp]
!6 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !2, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]

; Extract out the file from the replaced subprogram.  Confirm that each
; subprogram is pointing at the correct function.
; CHECK-DAG: ![[SP1]] = !{!"0x2e{{[^"]+}}", {{.*}}, i32 ()* @_Z3foov,
; CHECK-DAG: ![[SP2]] = !{!"0x2e{{[^"]+}}", ![[FILE:[0-9]+]], {{.*}}, i32 (%struct.Class*)* @_ZN5ClassIiE3fooEv,
!7 = !{!"0x2e\00foo\00foo\00\002\000\001\000\000\00256\000\002", !8, !9, !6, null, i32 (%struct.Class*)* @_ZN5ClassIiE3fooEv, null, null, !2} ; [ DW_TAG_subprogram ] [line 2] [def] [foo]

; The new subprogram should be pointing at the new directory.
; CHECK-DAG: ![[FILE]] = !{!"../t.h", !"/Users/dexonsmith/data/llvm/staging/test/Linker/repro/d2"}
!8 = !{!"../t.h", !"/Users/dexonsmith/data/llvm/staging/test/Linker/repro/d1"}
!9 = !{!"0x29", !8}    ; [ DW_TAG_file_type ] [/Users/dexonsmith/data/llvm/staging/test/Linker/repro/d1/../t.h]
!10 = !{i32 2, !"Dwarf Version", i32 2}
!11 = !{i32 2, !"Debug Info Version", i32 2}
!12 = !{i32 1, !"PIC Level", i32 2}
!13 = !{!"clang version 3.6.0 (trunk 224193) (llvm/trunk 224197)"}
!14 = !MDLocation(line: 2, column: 20, scope: !4)

; The same subprogram should be pointed to by inside the !dbg reference.
; CHECK: ![[LOC]] = !MDLocation(line: 2, column: 15, scope: ![[SP2]])
!15 = !MDLocation(line: 2, column: 15, scope: !7)
