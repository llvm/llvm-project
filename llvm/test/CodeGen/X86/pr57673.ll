; RUN: llc -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -stop-after=x86-optimize-LEAs -experimental-debug-variable-locations=false < %s \
; RUN:   | FileCheck %s --check-prefix=NORMAL
; RUN: llc -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -stop-after=x86-optimize-LEAs -experimental-debug-variable-locations < %s \
; RUN:   | FileCheck %s --check-prefix=INSTRREF

; The LEA optimization pass used to crash on this testcase.

; This test case used to trigger:
;
;   assert(MRI->use_empty(LastVReg) &&
;          "The LEA's def register must have no uses");

; CHECK:     LEA64r
; CHECK-NOT: LEA64r
; NORMAL:    DBG_VALUE_LIST
; INSTRREF:  DBG_INSTR_REF

target triple = "x86_64-unknown-linux-gnu"

%t10 = type { ptr, [32 x i8] }

define void @foo() {
bb_entry:
  %tmp11 = alloca [0 x [0 x i32]], i32 0, align 4
  %i = alloca %t10, align 8
  %i1 = alloca %t10, align 8
  br label %bb_8

bb_8:                                             ; preds = %bb_last, %bb_entry
  br i1 undef, label %bb_last, label %bb_mid

bb_mid:                                           ; preds = %bb_8
  %i4 = getelementptr inbounds %t10, ptr %i1, i64 0, i32 1, i64 32
  %i6 = getelementptr inbounds %t10, ptr %i, i64 0, i32 1, i64 32
  call void @llvm.lifetime.start.p0(i64 0, ptr nonnull %i1)
  %v21 = call i64 @llvm.ctlz.i64(i64 undef, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull dereferenceable(16) null, ptr noundef nonnull align 8 dereferenceable(16) %i4, i64 16, i1 false)
  call void @llvm.dbg.value(metadata !DIArgList(ptr %i4, ptr %i4), metadata !4, metadata !DIExpression(DW_OP_LLVM_arg, 0)), !dbg !9
  call void @llvm.lifetime.end.p0(i64 0, ptr nonnull %i1)
  call void @llvm.lifetime.start.p0(i64 0, ptr nonnull %i)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull dereferenceable(16) null, ptr noundef nonnull align 8 dereferenceable(16) %i6, i64 16, i1 false)
  call void @llvm.lifetime.end.p0(i64 0, ptr nonnull %i)
  br label %bb_last

bb_last:                                          ; preds = %bb_mid, %bb_8
  call void @llvm.lifetime.start.p0(i64 0, ptr nonnull %tmp11)
  call void undef(ptr null, ptr null, ptr null, i32 0, ptr nonnull %tmp11)
  call void @llvm.lifetime.end.p0(i64 0, ptr nonnull %tmp11)
  br label %bb_8
}

declare i64 @llvm.ctlz.i64(i64, i1 immarg)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug)
!1 = !DIFile(filename: "n", directory: "/proc/self/cwd", checksumkind: CSK_MD5, checksum: "e588179fedd8fcdfada963f2434cb950")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{!"function_entry_count", i64 2423}
!4 = !DILocalVariable(name: "r", scope: !5, file: !6, line: 93)
!5 = distinct !DISubprogram(name: "c", scope: !7, file: !6, line: 92, spFlags: DISPFlagDefinition, unit: !0)
!6 = !DIFile(filename: "a", directory: "/proc/self/cwd")
!7 = !DINamespace(name: "u", scope: !8)
!8 = !DINamespace(name: "s", scope: null)
!9 = !DILocation(line: 0, scope: !5)
