; RUN: llc < %s -frame-pointer=all -mtriple x86_64-apple-darwin11 -mcpu corei7 -emit-compact-unwind-non-canonical=true | FileCheck -check-prefix=ASM %s
; RUN: llc < %s -frame-pointer=all -mtriple x86_64-apple-darwin11 -mcpu corei7 -filetype=obj -emit-compact-unwind-non-canonical=true -o - \
; RUN:  | llvm-objdump --triple=x86_64-apple-darwin11 --unwind-info - \
; RUN:  | FileCheck -check-prefix=CU %s
; RUN: llc < %s -frame-pointer=all -mtriple x86_64-apple-darwin11 -mcpu corei7 -emit-compact-unwind-non-canonical=true \
; RUN:  | llvm-mc -triple x86_64-apple-darwin11 -filetype=obj -emit-compact-unwind-non-canonical=true -o - \
; RUN:  | llvm-objdump --triple=x86_64-apple-darwin11 --unwind-info - \
; RUN:  | FileCheck -check-prefix=FROM-ASM %s

; RUN: llc < %s -mtriple x86_64-apple-macosx10.8.0 -mcpu corei7 -filetype=obj -emit-compact-unwind-non-canonical=true -o - \
; RUN:  | llvm-objdump --triple=x86_64-apple-macosx10.8.0 --unwind-info - \
; RUN:  | FileCheck -check-prefix=NOFP-CU %s
; RUN: llc < %s -mtriple x86_64-apple-darwin11 -mcpu corei7 \
; RUN:  | llvm-mc -triple x86_64-apple-darwin11 -filetype=obj -emit-compact-unwind-non-canonical=true -o - \
; RUN:  | llvm-objdump --triple=x86_64-apple-darwin11 --unwind-info - \
; RUN:  | FileCheck -check-prefix=NOFP-FROM-ASM %s

%ty = type { i8* }

@gv = external global i32

; This is aligning the stack with a push of a random register.
; ASM: pushq %rax

; Even though we can't encode %rax into the compact unwind, We still want to be
; able to generate a compact unwind encoding in this particular case.

; CU:    Contents of __compact_unwind section:
; CU-NEXT:      Entry at offset 0x0:
; CU-NEXT:        start:                0x0 _test0
; CU-NEXT:        length:               0x1e
; CU-NEXT:        compact encoding:     0x01010001

; FROM-ASM:    Contents of __compact_unwind section:
; FROM-ASM-NEXT:      Entry at offset 0x0:
; FROM-ASM-NEXT:        start:                0x0 _test0
; FROM-ASM-NEXT:        length:               0x1e
; FROM-ASM-NEXT:        compact encoding:     0x01010001

define i8* @test0(i64 %size) {
  %addr = alloca i64, align 8
  %tmp20 = load i32, i32* @gv, align 4
  %tmp21 = call i32 @bar()
  %tmp25 = load i64, i64* %addr, align 8
  %tmp26 = inttoptr i64 %tmp25 to %ty*
  %tmp29 = getelementptr inbounds %ty, %ty* %tmp26, i64 0, i32 0
  %tmp34 = load i8*, i8** %tmp29, align 8
  %tmp35 = getelementptr inbounds i8, i8* %tmp34, i64 %size
  store i8* %tmp35, i8** %tmp29, align 8
  ret i8* null
}

declare i32 @bar()

%"struct.dyld::MappedRanges" = type { [400 x %struct.anon], %"struct.dyld::MappedRanges"* }
%struct.anon = type { %class.ImageLoader*, i64, i64 }
%class.ImageLoader = type { i32 (...)**, i8*, i8*, i32, i64, i64, i32, i32, %"struct.ImageLoader::recursive_lock"*, i16, i16, [4 x i8] }
%"struct.ImageLoader::recursive_lock" = type { i32, i32 }

@G1 = external hidden global %"struct.dyld::MappedRanges", align 8

declare void @OSMemoryBarrier() optsize

; Test the code below uses UNWIND_X86_64_MODE_STACK_IMMD compact unwind
; encoding.

; NOFP-CU:      Entry at offset 0x20:
; NOFP-CU-NEXT:        start:                0x1d _test1
; NOFP-CU-NEXT:        length:               0x44
; NOFP-CU-NEXT:        compact encoding:     0x02040c0a

; NOFP-FROM-ASM:      Entry at offset 0x20:
; NOFP-FROM-ASM-NEXT:        start:                0x1d _test1
; NOFP-FROM-ASM-NEXT:        length:               0x44
; NOFP-FROM-ASM-NEXT:        compact encoding:     0x02040c0a

define void @test1(%class.ImageLoader* %image) optsize ssp uwtable {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.inc10, %entry
  %p.019 = phi %"struct.dyld::MappedRanges"* [ @G1, %entry ], [ %1, %for.inc10 ]
  br label %for.body3

for.body3:                                        ; preds = %for.inc, %for.cond1.preheader
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.inc ]
  %image4 = getelementptr inbounds %"struct.dyld::MappedRanges", %"struct.dyld::MappedRanges"* %p.019, i64 0, i32 0, i64 %indvars.iv, i32 0
  %0 = load %class.ImageLoader*, %class.ImageLoader** %image4, align 8
  %cmp5 = icmp eq %class.ImageLoader* %0, %image
  br i1 %cmp5, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body3
  tail call void @OSMemoryBarrier() optsize
  store %class.ImageLoader* null, %class.ImageLoader** %image4, align 8
  br label %for.inc

for.inc:                                          ; preds = %if.then, %for.body3
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 400
  br i1 %exitcond, label %for.inc10, label %for.body3

for.inc10:                                        ; preds = %for.inc
  %next = getelementptr inbounds %"struct.dyld::MappedRanges", %"struct.dyld::MappedRanges"* %p.019, i64 0, i32 1
  %1 = load %"struct.dyld::MappedRanges"*, %"struct.dyld::MappedRanges"** %next, align 8
  %cmp = icmp eq %"struct.dyld::MappedRanges"* %1, null
  br i1 %cmp, label %for.end11, label %for.cond1.preheader

for.end11:                                        ; preds = %for.inc10
  ret void
}
