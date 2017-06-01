; RUN: opt < %s -instcombine -disable-output
; Checks that bitcasts are not converted into GEP when
; when the size of an aggregate cannot be determined.
%swift.opaque = type opaque
%SQ = type <{ [8 x i8] }>
%Si = type <{ i64 }>

%V = type <{ <{ %Vs4Int8, %Vs4Int8, %Vs4Int8, %Vs4Int8, %Vs4Int8, %Vs4Int8, %Vs4Int8, %Vs4Int8 }>, %Si, %SQ, %SQ, %Si, %swift.opaque }>
%Vs4Int8 = type <{ i8 }>
%swift.type = type { i64 }

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly,
i64, i32, i1) #8

@_swift_slowAlloc = external global i8* (i64, i64)*

declare i8* @rt_swift_slowAlloc(i64, i64)

define  %swift.opaque* @_TwTkV([24 x i8]* %dest, %swift.opaque* %src,
%swift.type* %bios_boot_params) #0 {
entry:
  %0 = bitcast %swift.opaque* %src to %V*
  %1 = call noalias i8* @rt_swift_slowAlloc(i64 40, i64 0) #11
  %2 = bitcast [24 x i8]* %dest to i8**
  store i8* %1, i8** %2, align 8
  %3 = bitcast i8* %1 to %V*
  %4 = bitcast %V* %3 to i8*
  %5 = bitcast %V* %0 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %4, i8* %5, i64 40, i32 1, i1 false)
  %6 = bitcast %V* %3 to %swift.opaque*
  ret %swift.opaque* %6
}
