; RUN: llc -mtriple=x86_64-pc-windows-msvc -stop-after=prolog-epilog -o - %s -experimental-debug-variable-locations=true | FileCheck %s --check-prefix=WIN
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -stop-after=prolog-epilog -o - %s -experimental-debug-variable-locations=true | FileCheck %s --check-prefix=LINUX
; RUN: llc -mtriple=x86_64-pc-windows-msvc -filetype=obj -o - %s -experimental-debug-variable-locations=true | llvm-dwarfdump --name=c - | FileCheck %s --check-prefix=DWARF
; RUN: llc -mtriple=x86_64-pc-windows-msvc -filetype=obj -o - %s -experimental-debug-variable-locations=true | llvm-dwarfdump --name=g -c - | FileCheck %s --check-prefix=DWARF-D

; PrologEpilogInserter keeps the DBG_PHIs of argument registers and the
; parameter DBG_INSTR_REFs referring to them ahead of the prologue code, unless
; the prologue code modifies the register. With __regcall, 'c' is passed in
; EAX, which the Win64 stack probe clobbers to pass the frame size to __chkstk
; and reloads afterwards. LiveDebugValues cannot tell that the reload restores
; the value, so keep the DBG_PHI and DBG_INSTR_REF after the prologue code, or
; 'c' would lose its location from the clobber on.
;
;   void use(char *, _Bool);
;   void __regcall f(_Bool c) { char buf[8192]; use(buf, c); use(buf, c); }

; WIN-LABEL: name: f
; WIN:      bb.0.entry:
; WIN-NOT:    DBG_PHI
; WIN:        $rax = frame-setup MOV32ri64 8224
; WIN:        frame-setup CALL64pcrel32 &__chkstk
; WIN:        $rax = frame-setup MOV64rm $rsp
; WIN:        frame-setup SEH_EndPrologue
; WIN-NEXT:   DBG_PHI $eax, {{[0-9]+}}
; WIN-NEXT:   DBG_INSTR_REF !{{[0-9]+}}, !DIExpression(DW_OP_LLVM_arg, 0), dbg-instr-ref({{[0-9]+}}, 0)
; WIN-NEXT:   COPY $eax

; LINUX-LABEL: name: f
; LINUX:      bb.0.entry:
; LINUX-NOT:    frame-setup
; LINUX:        DBG_PHI $eax, {{[0-9]+}}
; LINUX-NEXT:   DBG_INSTR_REF !{{[0-9]+}}, !DIExpression(DW_OP_LLVM_arg, 0), dbg-instr-ref({{[0-9]+}}, 0)
; LINUX-NEXT:   frame-setup PUSH64r

; DWARF:      DW_AT_location
; DWARF-NEXT: [0x{{[0-9a-f]+}}, 0x[[END:[0-9a-f]+]]): DW_OP_reg0 RAX
; DWARF-NEXT: [0x[[END]], 0x{{[0-9a-f]+}}): DW_OP_reg4 RSI)
; DWARF:      DW_AT_name ("c")

;; Only the DBG_PHI of the clobbered register, and the DBG_INSTR_REF referring
;; to it, are kept after the prologue code: 'd' is passed in ECX, which the
;; prologue code does not modify, so it is still described from the entry on.
; WIN-LABEL: name: g
; WIN:      bb.0.entry:
; WIN-NEXT:   liveins:
; WIN-NOT:    frame-setup
; WIN:        DBG_PHI $ecx, {{[0-9]+}}
; WIN-NEXT:   DBG_INSTR_REF ![[D:[0-9]+]], !DIExpression(DW_OP_LLVM_arg, 0)
; WIN-NEXT:   frame-setup PUSH64r
; WIN:        frame-setup CALL64pcrel32 &__chkstk
; WIN:        frame-setup SEH_EndPrologue
; WIN-NEXT:   DBG_PHI $eax, {{[0-9]+}}
; WIN-NEXT:   DBG_INSTR_REF ![[C:[0-9]+]], !DIExpression(DW_OP_LLVM_arg, 0)
; WIN-NOT:    DBG_PHI
; WIN-NOT:    DBG_INSTR_REF ![[D]]

; DWARF-D:      DW_AT_low_pc (0x[[ENTRY:[0-9a-f]+]])
; DWARF-D:      DW_AT_name ("g")
; DWARF-D:      DW_AT_location
; DWARF-D:      DW_AT_name ("c")
; DWARF-D:      DW_AT_location
; DWARF-D-NEXT: [0x[[ENTRY]], {{.*}}): DW_OP_reg2 RCX
; DWARF-D:      DW_AT_name ("d")

declare void @use(ptr, i8)
declare void @use2(ptr, i8, i8)

define x86_regcallcc void @f(i32 %c) !dbg !7 {
entry:
  %buf = alloca [8192 x i8]
  %frombool = trunc i32 %c to i8
    #dbg_value(i8 %frombool, !13, !DIExpression(), !15)
  call void @use(ptr %buf, i8 %frombool), !dbg !16
  call void @use(ptr %buf, i8 %frombool), !dbg !16
  ret void, !dbg !16
}

define x86_regcallcc void @g(i32 %c, i32 %d) !dbg !20 {
entry:
  %buf = alloca [8192 x i8]
  %fc = trunc i32 %c to i8
    #dbg_value(i8 %fc, !21, !DIExpression(), !23)
  %fd = trunc i32 %d to i8
    #dbg_value(i8 %fd, !22, !DIExpression(), !23)
  call void @use2(ptr %buf, i8 %fc, i8 %fd), !dbg !24
  call void @use2(ptr %buf, i8 %fc, i8 %fd), !dbg !24
  ret void, !dbg !24
}
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!1 = !DIFile(filename: "t.c", directory: "")
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !DIBasicType(name: "_Bool", size: 8, encoding: DW_ATE_boolean)
!7 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !5}
!13 = !DILocalVariable(name: "c", arg: 1, scope: !7, file: !1, line: 2, type: !5)
!15 = !DILocation(line: 0, scope: !7)
!16 = !DILocation(line: 2, column: 40, scope: !7)
!20 = distinct !DISubprogram(name: "g", scope: !1, file: !1, line: 3, type: !8, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!21 = !DILocalVariable(name: "c", arg: 1, scope: !20, file: !1, line: 3, type: !5)
!22 = !DILocalVariable(name: "d", arg: 2, scope: !20, file: !1, line: 3, type: !5)
!23 = !DILocation(line: 0, scope: !20)
!24 = !DILocation(line: 3, column: 40, scope: !20)
