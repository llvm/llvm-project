; RUN: opt < %s -passes=instcombine -disable-output
; Checks that bitcasts are not converted into GEP when
; when the size of an aggregate cannot be determined.
%swift.opaque = type opaque
%SQ = type <{ [8 x i8] }>
%Si = type <{ i64 }>

%V = type <{ <{ %Vs4Int8, %Vs4Int8, %Vs4Int8, %Vs4Int8, %Vs4Int8, %Vs4Int8, %Vs4Int8, %Vs4Int8 }>, %Si, %SQ, %SQ, %Si, %swift.opaque }>
%Vs4Int8 = type <{ i8 }>
%swift.type = type { i64 }

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture readonly, i64, i1) #8

@_swift_slowAlloc = external global ptr

declare ptr @rt_swift_slowAlloc(i64, i64)

define  ptr @_TwTkV(ptr %dest, ptr %src,
ptr %bios_boot_params) #0 {
entry:
  %0 = call noalias ptr @rt_swift_slowAlloc(i64 40, i64 0) #11
  store ptr %0, ptr %dest, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr %0, ptr %src, i64 40, i1 false)
  ret ptr %0
}
