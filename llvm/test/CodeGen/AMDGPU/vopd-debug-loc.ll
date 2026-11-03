; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa -filetype=obj < %s \
; RUN:   | llvm-dwarfdump --debug-line - | FileCheck %s --check-prefix=LINES

; A VOPD instruction has the location of its X component, or of its Y component
; if X has none. If both have different locations, Y's is emitted to the line
; table right before the instruction's own, so both source lines are attributed
; to the instruction's address and X's row comes last.

; CHECK-LABEL: different_lines:
; CHECK:       .loc 0 20 5
; CHECK-NEXT:  {{^}}.Ltmp{{[0-9]+}}:
; CHECK-NEXT:  .loc 0 10 3 prologue_end
; CHECK-NEXT:  v_dual_mul_f32 {{.*}} :: v_dual_add_f32

; LINES:      [[ADDR:0x[0-9a-f]+]] 20 5 0 0 0 0 is_stmt{{$}}
; LINES-NEXT: [[ADDR]] 10 3 0 0 0 0 is_stmt prologue_end
define <2 x float> @different_lines(float %a, float %b, float %c, float %d) !dbg !5 {
  %x = fmul float %a, %b, !dbg !8
  %y = fadd float %c, %d, !dbg !9
  %v0 = insertelement <2 x float> poison, float %x, i32 0, !dbg !9
  %v1 = insertelement <2 x float> %v0, float %y, i32 1, !dbg !9
  ret <2 x float> %v1, !dbg !9
}

; Both components have the same location; nothing extra is emitted.
; CHECK-LABEL: same_line:
; CHECK:       .loc 0 30 0
; CHECK-NOT:   .loc
; CHECK:       .loc 0 31 7 prologue_end
; CHECK-NEXT:  v_dual_mul_f32 {{.*}} :: v_dual_add_f32
define <2 x float> @same_line(float %a, float %b, float %c, float %d) !dbg !10 {
  %x = fmul float %a, %b, !dbg !11
  %y = fadd float %c, %d, !dbg !11
  %v0 = insertelement <2 x float> poison, float %x, i32 0, !dbg !11
  %v1 = insertelement <2 x float> %v0, float %y, i32 1, !dbg !11
  ret <2 x float> %v1, !dbg !11
}

; X has no location, so the VOPD takes Y's location.
; CHECK-LABEL: x_without_loc:
; CHECK:       .loc 0 42 9
; CHECK-NEXT:  v_dual_mul_f32 {{.*}} :: v_dual_add_f32
define <2 x float> @x_without_loc(float %a, float %b, float %c, float %d) !dbg !12 {
  %x = fmul float %a, %b
  %y = fadd float %c, %d, !dbg !13
  %v0 = insertelement <2 x float> poison, float %x, i32 0
  %v1 = insertelement <2 x float> %v0, float %y, i32 1
  ret <2 x float> %v1, !dbg !14
}

; X's location has line 0, which attributes it to no line, so the VOPD takes
; Y's location.
; CHECK-LABEL: x_line_zero:
; CHECK:       .loc 0 102 5
; CHECK-NEXT:  v_dual_mul_f32 {{.*}} :: v_dual_add_f32
define <2 x float> @x_line_zero(float %a, float %b, float %c, float %d) !dbg !70 {
  %x = fmul float %a, %b, !dbg !71
  %y = fadd float %c, %d, !dbg !72
  %v0 = insertelement <2 x float> poison, float %x, i32 0, !dbg !73
  %v1 = insertelement <2 x float> %v0, float %y, i32 1, !dbg !73
  ret <2 x float> %v1, !dbg !73
}

; X is on the same line as the previous instruction. Its location is emitted
; again after Y's, and the next instruction on X's line gets no new row.
; CHECK-LABEL: prev_same_as_x:
; CHECK:       .loc 0 51 3 prologue_end
; CHECK-NEXT:  v_mul_f32_e32
; CHECK-NOT:   .loc
; CHECK:       .loc 0 52 5
; CHECK-NEXT:  .loc 0 51 3
; CHECK-NEXT:  v_dual_mul_f32 {{.*}} :: v_dual_add_f32
; CHECK-NEXT:  v_mul_f32_e32
; CHECK-NEXT:  .loc 0 53 1

; LINES:      [[ADDR:0x[0-9a-f]+]] 52 5 0 0 0 0 is_stmt{{$}}
; LINES-NEXT: [[ADDR]] 51 3 0 0 0 0 is_stmt{{$}}
define float @prev_same_as_x(float %a, float %b, float %c, float %d) !dbg !20 {
  %t = fmul float %a, %b, !dbg !21
  %x = fmul float %t, %c, !dbg !21
  %y = fadd float %t, %d, !dbg !22
  %z = fmul float %x, %y, !dbg !21
  ret float %z, !dbg !23
}

; Y is on the same line as the previous instruction. Its location is emitted
; again, so that both lines precede the VOPD, but not as a new statement.
; CHECK-LABEL: prev_same_as_y:
; CHECK:       .loc 0 82 5 prologue_end
; CHECK-NEXT:  v_add_f32_e32
; CHECK-NOT:   .loc
; CHECK:       .loc 0 82 5 is_stmt 0
; CHECK-NEXT:  .loc 0 81 3 is_stmt 1
; CHECK-NEXT:  v_dual_mul_f32 {{.*}} :: v_dual_add_f32
define float @prev_same_as_y(float %a, float %b, float %c, float %d) !dbg !50 {
  %t = fadd float %a, %b, !dbg !52
  %x = fmul float %t, %c, !dbg !51
  %y = fadd float %t, %d, !dbg !52
  %z = fmul float %x, %y, !dbg !53
  ret float %z, !dbg !53
}

; X has no location and the VOPD starts a block, where an instruction without a
; location would get a line-0 row. The VOPD takes Y's location instead.
; CHECK-LABEL: x_without_loc_block_start:
; CHECK:       %then
; CHECK-NEXT:  .loc 0 62 5
; CHECK-NEXT:  v_dual_mul_f32 {{.*}} :: v_dual_add_f32
define void @x_without_loc_block_start(ptr addrspace(1) %p, float %a, float %b, float %c, float %d, i32 inreg %n) !dbg !30 {
entry:
  %cond = icmp eq i32 %n, 0, !dbg !31
  br i1 %cond, label %then, label %exit, !dbg !31

then:
  %x = fmul float %a, %b
  %y = fadd float %c, %d, !dbg !32
  %v0 = insertelement <2 x float> poison, float %x, i32 0, !dbg !33
  %v1 = insertelement <2 x float> %v0, float %y, i32 1, !dbg !33
  store <2 x float> %v1, ptr addrspace(1) %p, align 8, !dbg !33
  br label %exit, !dbg !33

exit:
  ret void, !dbg !34
}

; The scalar move materializing the pair's shared literal takes the pair's
; location.
; CHECK-LABEL: literal_move:
; CHECK:       .loc 0 91 3 prologue_end
; CHECK-NEXT:  s_mov_b32 [[SREG:s[0-9]+]], 0xffff0000
; CHECK-NOT:   .loc
; CHECK:       .loc 0 92 5
; CHECK-NEXT:  .loc 0 91 3
; CHECK-NEXT:  v_dual_lshlrev_b32 {{.*}} :: v_dual_bitop2_b32 {{.*}}, [[SREG]],
define <2 x i32> @literal_move(i32 %a, i32 %b) !dbg !60 {
  %x = shl i32 %a, 16, !dbg !61
  %y = and i32 %b, -65536, !dbg !62
  %v0 = insertelement <2 x i32> poison, i32 %x, i32 0, !dbg !63
  %v1 = insertelement <2 x i32> %v0, i32 %y, i32 1, !dbg !63
  ret <2 x i32> %v1, !dbg !63
}

; With Key Instructions, Y's row is not a statement even though Y was the key
; instruction of its atom.
; FIXME: Y's row should have is_stmt.
; CHECK-LABEL: key_instructions:
; CHECK:       .loc 0 72 5 is_stmt 0
; CHECK-NEXT:  {{^}}.Ltmp{{[0-9]+}}:
; CHECK-NEXT:  .loc 0 71 3 prologue_end is_stmt 1
; CHECK-NEXT:  v_dual_mul_f32 {{.*}} :: v_dual_add_f32
define <2 x float> @key_instructions(float %a, float %b, float %c, float %d) !dbg !40 {
  %x = fmul float %a, %b, !dbg !41
  %y = fadd float %c, %d, !dbg !42
  %v0 = insertelement <2 x float> poison, float %x, i32 0, !dbg !43
  %v1 = insertelement <2 x float> %v0, float %y, i32 1, !dbg !43
  ret <2 x float> %v1, !dbg !43
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !DISubroutineType(types: !{})
!5 = distinct !DISubprogram(name: "different_lines", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!8 = !DILocation(line: 10, column: 3, scope: !5)
!9 = !DILocation(line: 20, column: 5, scope: !5)
!10 = distinct !DISubprogram(name: "same_line", scope: !1, file: !1, line: 30, type: !4, scopeLine: 30, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = !DILocation(line: 31, column: 7, scope: !10)
!12 = distinct !DISubprogram(name: "x_without_loc", scope: !1, file: !1, line: 40, type: !4, scopeLine: 40, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!13 = !DILocation(line: 42, column: 9, scope: !12)
!14 = !DILocation(line: 43, column: 1, scope: !12)
!20 = distinct !DISubprogram(name: "prev_same_as_x", scope: !1, file: !1, line: 50, type: !4, scopeLine: 50, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!21 = !DILocation(line: 51, column: 3, scope: !20)
!22 = !DILocation(line: 52, column: 5, scope: !20)
!23 = !DILocation(line: 53, column: 1, scope: !20)
!30 = distinct !DISubprogram(name: "x_without_loc_block_start", scope: !1, file: !1, line: 60, type: !4, scopeLine: 60, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!31 = !DILocation(line: 61, column: 3, scope: !30)
!32 = !DILocation(line: 62, column: 5, scope: !30)
!33 = !DILocation(line: 63, column: 7, scope: !30)
!34 = !DILocation(line: 64, column: 1, scope: !30)
!40 = distinct !DISubprogram(name: "key_instructions", scope: !1, file: !1, line: 70, type: !4, scopeLine: 70, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, keyInstructions: true)
!41 = !DILocation(line: 71, column: 3, scope: !40, atomGroup: 1, atomRank: 1)
!42 = !DILocation(line: 72, column: 5, scope: !40, atomGroup: 2, atomRank: 1)
!43 = !DILocation(line: 73, column: 1, scope: !40, atomGroup: 3, atomRank: 1)
!50 = distinct !DISubprogram(name: "prev_same_as_y", scope: !1, file: !1, line: 80, type: !4, scopeLine: 80, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!51 = !DILocation(line: 81, column: 3, scope: !50)
!52 = !DILocation(line: 82, column: 5, scope: !50)
!53 = !DILocation(line: 83, column: 1, scope: !50)
!60 = distinct !DISubprogram(name: "literal_move", scope: !1, file: !1, line: 90, type: !4, scopeLine: 90, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!61 = !DILocation(line: 91, column: 3, scope: !60)
!62 = !DILocation(line: 92, column: 5, scope: !60)
!63 = !DILocation(line: 93, column: 1, scope: !60)
!70 = distinct !DISubprogram(name: "x_line_zero", scope: !1, file: !1, line: 100, type: !4, scopeLine: 100, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!71 = !DILocation(line: 0, scope: !70)
!72 = !DILocation(line: 102, column: 5, scope: !70)
!73 = !DILocation(line: 103, column: 1, scope: !70)
