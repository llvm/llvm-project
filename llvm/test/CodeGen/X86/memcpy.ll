; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=core2 | FileCheck %s -check-prefix=LINUX
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core2 | FileCheck %s -check-prefix=DARWIN

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind
declare void @llvm.memcpy.p256i8.p256i8.i64(i8 addrspace(256)* nocapture, i8 addrspace(256)* nocapture, i64, i32, i1) nounwind


; Variable memcpy's should lower to calls.
define i8* @test1(i8* %a, i8* %b, i64 %n) nounwind {
entry:
	tail call void @llvm.memcpy.p0i8.p0i8.i64( i8* %a, i8* %b, i64 %n, i32 1, i1 0 )
	ret i8* %a
        
; LINUX-LABEL: test1:
; LINUX: memcpy
}

; Variable memcpy's should lower to calls.
define i8* @test2(i64* %a, i64* %b, i64 %n) nounwind {
entry:
	%tmp14 = bitcast i64* %a to i8*
	%tmp25 = bitcast i64* %b to i8*
	tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %tmp14, i8* %tmp25, i64 %n, i32 8, i1 0 )
	ret i8* %tmp14
        
; LINUX-LABEL: test2:
; LINUX: memcpy
}

; Large constant memcpy's should lower to a call when optimizing for size.
; PR6623

; On the other hand, Darwin's definition of -Os is optimizing for size without
; hurting performance so it should just ignore optsize when expanding memcpy.
; rdar://8821501
define void @test3(i8* nocapture %A, i8* nocapture %B) nounwind optsize noredzone {
entry:
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %A, i8* %B, i64 64, i32 1, i1 false)
  ret void
; LINUX-LABEL: test3:
; LINUX: memcpy

; DARWIN-LABEL: test3:
; DARWIN-NOT: memcpy
; DARWIN: movq
; DARWIN: movq
; DARWIN: movq
; DARWIN: movq
; DARWIN: movq
; DARWIN: movq
; DARWIN: movq
; DARWIN: movq
; DARWIN: movq
; DARWIN: movq
; DARWIN: movq
; DARWIN: movq
; DARWIN: movq
; DARWIN: movq
; DARWIN: movq
; DARWIN: movq
}

define void @test3_minsize(i8* nocapture %A, i8* nocapture %B) nounwind minsize noredzone {
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %A, i8* %B, i64 64, i32 1, i1 false)
  ret void
; LINUX-LABEL: test3_minsize:
; LINUX: memcpy

; DARWIN-LABEL: test3_minsize:
; DARWIN: memcpy
}

define void @test3_minsize_optsize(i8* nocapture %A, i8* nocapture %B) nounwind optsize minsize noredzone {
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %A, i8* %B, i64 64, i32 1, i1 false)
  ret void
; LINUX-LABEL: test3_minsize_optsize:
; LINUX: memcpy

; DARWIN-LABEL: test3_minsize_optsize:
; DARWIN: memcpy
}

; Large constant memcpy's should be inlined when not optimizing for size.
define void @test4(i8* nocapture %A, i8* nocapture %B) nounwind noredzone {
entry:
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %A, i8* %B, i64 64, i32 1, i1 false)
  ret void
; LINUX-LABEL: test4:
; LINUX: movq
; LINUX: movq
; LINUX: movq
; LINUX: movq
; LINUX: movq
; LINUX: movq
; LINUX: movq
; LINUX: movq
; LINUX: movq
; LINUX: movq
; LINUX: movq
; LINUX: movq
}


@.str = private unnamed_addr constant [30 x i8] c"\00aaaaaaaaaaaaaaaaaaaaaaaaaaaa\00", align 1

define void @test5(i8* nocapture %C) nounwind uwtable ssp {
entry:
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %C, i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str, i64 0, i64 0), i64 16, i32 1, i1 false)
  ret void

; DARWIN-LABEL: test5:
; DARWIN: movabsq	$7016996765293437281
; DARWIN: movabsq	$7016996765293437184
}


; PR14896
@.str2 = private unnamed_addr constant [2 x i8] c"x\00", align 1

define void @test6() nounwind uwtable {
entry:
; DARWIN: test6
; DARWIN: movw $0, 8
; DARWIN: movq $120, 0
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* null, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str2, i64 0, i64 0), i64 10, i32 1, i1 false)
  ret void
}

define void @PR15348(i8* %a, i8* %b) {
; Ensure that alignment of '0' in an @llvm.memcpy intrinsic results in
; unaligned loads and stores.
; LINUX: PR15348
; LINUX: movb
; LINUX: movb
; LINUX: movq
; LINUX: movq
; LINUX: movq
; LINUX: movq
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %a, i8* %b, i64 17, i32 0, i1 false)
  ret void
}

; Memcpys from / to address space 256 should be lowered to appropriate loads /
; stores if small enough.
define void @addrspace256(i8 addrspace(256)* %a, i8 addrspace(256)* %b) nounwind {
  tail call void @llvm.memcpy.p256i8.p256i8.i64(i8 addrspace(256)* %a, i8 addrspace(256)* %b, i64 16, i32 8, i1 false)
  ret void
; LINUX-LABEL: addrspace256:
; LINUX: movq %gs:
; LINUX: movq %gs:
; LINUX: movq {{.*}}, %gs:
; LINUX: movq {{.*}}, %gs:
}
