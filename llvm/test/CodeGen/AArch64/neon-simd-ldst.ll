; RUN: llc < %s -O2 -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s

define void @test_ldstq_4v(i8* noalias %io, i32 %count) {
; CHECK-LABEL: test_ldstq_4v
; CHECK: ld4     {v0.16b, v1.16b, v2.16b, v3.16b}, [x0]
; CHECK: st4     {v0.16b, v1.16b, v2.16b, v3.16b}, [x0]
entry:
  %tobool62 = icmp eq i32 %count, 0
  br i1 %tobool62, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %count.addr.063 = phi i32 [ %dec, %while.body ], [ %count, %entry ]
  %dec = add i32 %count.addr.063, -1
  %vld4 = tail call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.arm.neon.vld4.v16i8(i8* %io, i32 1)
  %vld4.fca.0.extract = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %vld4, 0
  %vld4.fca.1.extract = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %vld4, 1
  %vld4.fca.2.extract = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %vld4, 2
  %vld4.fca.3.extract = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %vld4, 3
  tail call void @llvm.arm.neon.vst4.v16i8(i8* %io, <16 x i8> %vld4.fca.0.extract, <16 x i8> %vld4.fca.1.extract, <16 x i8> %vld4.fca.2.extract, <16 x i8> %vld4.fca.3.extract, i32 1)
  %tobool = icmp eq i32 %dec, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  ret void
}

declare { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.arm.neon.vld4.v16i8(i8*, i32)

declare void @llvm.arm.neon.vst4.v16i8(i8*, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, i32)

define void @test_ldstq_3v(i8* noalias %io, i32 %count) {
; CHECK-LABEL: test_ldstq_3v
; CHECK: ld3     {v0.16b, v1.16b, v2.16b}, [x0]
; CHECK: st3     {v0.16b, v1.16b, v2.16b}, [x0]
entry:
  %tobool47 = icmp eq i32 %count, 0
  br i1 %tobool47, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %count.addr.048 = phi i32 [ %dec, %while.body ], [ %count, %entry ]
  %dec = add i32 %count.addr.048, -1
  %vld3 = tail call { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.arm.neon.vld3.v16i8(i8* %io, i32 1)
  %vld3.fca.0.extract = extractvalue { <16 x i8>, <16 x i8>, <16 x i8> } %vld3, 0
  %vld3.fca.1.extract = extractvalue { <16 x i8>, <16 x i8>, <16 x i8> } %vld3, 1
  %vld3.fca.2.extract = extractvalue { <16 x i8>, <16 x i8>, <16 x i8> } %vld3, 2
  tail call void @llvm.arm.neon.vst3.v16i8(i8* %io, <16 x i8> %vld3.fca.0.extract, <16 x i8> %vld3.fca.1.extract, <16 x i8> %vld3.fca.2.extract, i32 1)
  %tobool = icmp eq i32 %dec, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  ret void
}

declare { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.arm.neon.vld3.v16i8(i8*, i32)

declare void @llvm.arm.neon.vst3.v16i8(i8*, <16 x i8>, <16 x i8>, <16 x i8>, i32)

define void @test_ldstq_2v(i8* noalias %io, i32 %count) {
; CHECK-LABEL: test_ldstq_2v
; CHECK: ld2     {v0.16b, v1.16b}, [x0]
; CHECK: st2     {v0.16b, v1.16b}, [x0]
entry:
  %tobool22 = icmp eq i32 %count, 0
  br i1 %tobool22, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %count.addr.023 = phi i32 [ %dec, %while.body ], [ %count, %entry ]
  %dec = add i32 %count.addr.023, -1
  %vld2 = tail call { <16 x i8>, <16 x i8> } @llvm.arm.neon.vld2.v16i8(i8* %io, i32 1)
  %vld2.fca.0.extract = extractvalue { <16 x i8>, <16 x i8> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <16 x i8>, <16 x i8> } %vld2, 1
  tail call void @llvm.arm.neon.vst2.v16i8(i8* %io, <16 x i8> %vld2.fca.0.extract, <16 x i8> %vld2.fca.1.extract, i32 1)
  %tobool = icmp eq i32 %dec, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  ret void
}

declare { <16 x i8>, <16 x i8> } @llvm.arm.neon.vld2.v16i8(i8*, i32)

declare void @llvm.arm.neon.vst2.v16i8(i8*, <16 x i8>, <16 x i8>, i32)

define void @test_ldst_4v(i8* noalias %io, i32 %count) {
; CHECK-LABEL: test_ldst_4v
; CHECK: ld4     {v0.8b, v1.8b, v2.8b, v3.8b}, [x0]
; CHECK: st4     {v0.8b, v1.8b, v2.8b, v3.8b}, [x0]
entry:
  %tobool42 = icmp eq i32 %count, 0
  br i1 %tobool42, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %count.addr.043 = phi i32 [ %dec, %while.body ], [ %count, %entry ]
  %dec = add i32 %count.addr.043, -1
  %vld4 = tail call { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.arm.neon.vld4.v8i8(i8* %io, i32 1)
  %vld4.fca.0.extract = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %vld4, 0
  %vld4.fca.1.extract = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %vld4, 1
  %vld4.fca.2.extract = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %vld4, 2
  %vld4.fca.3.extract = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %vld4, 3
  tail call void @llvm.arm.neon.vst4.v8i8(i8* %io, <8 x i8> %vld4.fca.0.extract, <8 x i8> %vld4.fca.1.extract, <8 x i8> %vld4.fca.2.extract, <8 x i8> %vld4.fca.3.extract, i32 1)
  %tobool = icmp eq i32 %dec, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  ret void
}

declare { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.arm.neon.vld4.v8i8(i8*, i32)

declare void @llvm.arm.neon.vst4.v8i8(i8*, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, i32)

define void @test_ldst_3v(i8* noalias %io, i32 %count) {
; CHECK-LABEL: test_ldst_3v
; CHECK: ld3     {v0.8b, v1.8b, v2.8b}, [x0]
; CHECK: st3     {v0.8b, v1.8b, v2.8b}, [x0]
entry:
  %tobool32 = icmp eq i32 %count, 0
  br i1 %tobool32, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %count.addr.033 = phi i32 [ %dec, %while.body ], [ %count, %entry ]
  %dec = add i32 %count.addr.033, -1
  %vld3 = tail call { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.arm.neon.vld3.v8i8(i8* %io, i32 1)
  %vld3.fca.0.extract = extractvalue { <8 x i8>, <8 x i8>, <8 x i8> } %vld3, 0
  %vld3.fca.1.extract = extractvalue { <8 x i8>, <8 x i8>, <8 x i8> } %vld3, 1
  %vld3.fca.2.extract = extractvalue { <8 x i8>, <8 x i8>, <8 x i8> } %vld3, 2
  tail call void @llvm.arm.neon.vst3.v8i8(i8* %io, <8 x i8> %vld3.fca.0.extract, <8 x i8> %vld3.fca.1.extract, <8 x i8> %vld3.fca.2.extract, i32 1)
  %tobool = icmp eq i32 %dec, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  ret void
}

declare { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.arm.neon.vld3.v8i8(i8*, i32)

declare void @llvm.arm.neon.vst3.v8i8(i8*, <8 x i8>, <8 x i8>, <8 x i8>, i32)

define void @test_ldst_2v(i8* noalias %io, i32 %count) {
; CHECK-LABEL: test_ldst_2v
; CHECK: ld2     {v0.8b, v1.8b}, [x0]
; CHECK: st2     {v0.8b, v1.8b}, [x0]
entry:
  %tobool22 = icmp eq i32 %count, 0
  br i1 %tobool22, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %count.addr.023 = phi i32 [ %dec, %while.body ], [ %count, %entry ]
  %dec = add i32 %count.addr.023, -1
  %vld2 = tail call { <8 x i8>, <8 x i8> } @llvm.arm.neon.vld2.v8i8(i8* %io, i32 1)
  %vld2.fca.0.extract = extractvalue { <8 x i8>, <8 x i8> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <8 x i8>, <8 x i8> } %vld2, 1
  tail call void @llvm.arm.neon.vst2.v8i8(i8* %io, <8 x i8> %vld2.fca.0.extract, <8 x i8> %vld2.fca.1.extract, i32 1)
  %tobool = icmp eq i32 %dec, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  ret void
}

declare { <8 x i8>, <8 x i8> } @llvm.arm.neon.vld2.v8i8(i8*, i32)

declare void @llvm.arm.neon.vst2.v8i8(i8*, <8 x i8>, <8 x i8>, i32)

