; RUN: opt -O2 -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin11.1"

%"struct.boost::compressed_pair<empty_t,int>" = type { %"struct.boost::details::compressed_pair_imp<empty_t,int,1>" }
%"struct.boost::details::compressed_pair_imp<empty_t,int,1>" = type { i32 }
%struct.empty_base_t = type <{ i8 }>
%struct.empty_t = type <{ i8 }>

@.str = private constant [25 x i8] c"x.second() was clobbered\00", align 1 ; <ptr> [#uses=1]

define i32 @main(i32 %argc, ptr %argv) ssp {
entry:
  %argc_addr = alloca i32, align 4                ; <ptr> [#uses=1]
  %argv_addr = alloca ptr, align 8               ; <ptr> [#uses=1]
  %retval = alloca i32                            ; <ptr> [#uses=2]
  %0 = alloca i32                                 ; <ptr> [#uses=2]
  %retval.1 = alloca i8                           ; <ptr> [#uses=2]
  %1 = alloca %struct.empty_base_t                ; <ptr> [#uses=1]
  %2 = alloca ptr               ; <ptr> [#uses=1]
  %x = alloca %"struct.boost::compressed_pair<empty_t,int>" ; <ptr> [#uses=3]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  store i32 %argc, ptr %argc_addr
  store ptr %argv, ptr %argv_addr
  %3 = call ptr @_ZN5boost15compressed_pairI7empty_tiE6secondEv(ptr %x) ssp ; <ptr> [#uses=1]
  store i32 -3, ptr %3, align 4
  %4 = call ptr @_ZN5boost15compressed_pairI7empty_tiE5firstEv(ptr %x) ssp ; <ptr> [#uses=1]
  store ptr %4, ptr %2, align 8
  call void @_ZN7empty_tC1Ev(ptr %1) nounwind
  %5 = call ptr @_ZN5boost15compressed_pairI7empty_tiE6secondEv(ptr %x) ssp ; <ptr> [#uses=1]
  %6 = load i32, ptr %5, align 4                      ; <i32> [#uses=1]
  %7 = icmp ne i32 %6, -3                         ; <i1> [#uses=1]
  %8 = zext i1 %7 to i8                           ; <i8> [#uses=1]
  store i8 %8, ptr %retval.1, align 1
  %9 = load i8, ptr %retval.1, align 1                ; <i8> [#uses=1]
  %toBool = icmp ne i8 %9, 0                      ; <i1> [#uses=1]
  br i1 %toBool, label %bb, label %bb1

bb:                                               ; preds = %entry
  %10 = call i32 @puts(ptr @.str) ; <i32> [#uses=0]
  call void @abort() noreturn
  unreachable

bb1:                                              ; preds = %entry
  store i32 0, ptr %0, align 4
  %11 = load i32, ptr %0, align 4                     ; <i32> [#uses=1]
  store i32 %11, ptr %retval, align 4
  br label %return

; CHECK-NOT: x.second() was clobbered
; CHECK: ret i32
return:                                           ; preds = %bb1
  %retval2 = load i32, ptr %retval                    ; <i32> [#uses=1]
  ret i32 %retval2
}

define linkonce_odr void @_ZN12empty_base_tC2Ev(ptr %this) nounwind ssp align 2 {
entry:
  %this_addr = alloca ptr, align 8 ; <ptr> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  store ptr %this, ptr %this_addr
  br label %return

return:                                           ; preds = %entry
  ret void
}

define linkonce_odr void @_ZN7empty_tC1Ev(ptr %this) nounwind ssp align 2 {
entry:
  %this_addr = alloca ptr, align 8 ; <ptr> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  store ptr %this, ptr %this_addr
  %0 = load ptr, ptr %this_addr, align 8 ; <ptr> [#uses=1]
  call void @_ZN12empty_base_tC2Ev(ptr %0) nounwind
  br label %return

return:                                           ; preds = %entry
  ret void
}

define linkonce_odr ptr @_ZN5boost7details19compressed_pair_impI7empty_tiLi1EE6secondEv(ptr %this) nounwind ssp align 2 {
entry:
  %this_addr = alloca ptr, align 8 ; <ptr> [#uses=2]
  %retval = alloca ptr                           ; <ptr> [#uses=2]
  %0 = alloca ptr                                ; <ptr> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  store ptr %this, ptr %this_addr
  %1 = load ptr, ptr %this_addr, align 8 ; <ptr> [#uses=1]
  %2 = getelementptr inbounds %"struct.boost::details::compressed_pair_imp<empty_t,int,1>", ptr %1, i32 0, i32 0 ; <ptr> [#uses=1]
  store ptr %2, ptr %0, align 8
  %3 = load ptr, ptr %0, align 8                     ; <ptr> [#uses=1]
  store ptr %3, ptr %retval, align 8
  br label %return

return:                                           ; preds = %entry
  %retval1 = load ptr, ptr %retval                   ; <ptr> [#uses=1]
  ret ptr %retval1
}

define linkonce_odr ptr @_ZN5boost15compressed_pairI7empty_tiE6secondEv(ptr %this) ssp align 2 {
entry:
  %this_addr = alloca ptr, align 8 ; <ptr> [#uses=2]
  %retval = alloca ptr                           ; <ptr> [#uses=2]
  %0 = alloca ptr                                ; <ptr> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  store ptr %this, ptr %this_addr
  %1 = load ptr, ptr %this_addr, align 8 ; <ptr> [#uses=1]
  %2 = getelementptr inbounds %"struct.boost::compressed_pair<empty_t,int>", ptr %1, i32 0, i32 0 ; <ptr> [#uses=1]
  %3 = call ptr @_ZN5boost7details19compressed_pair_impI7empty_tiLi1EE6secondEv(ptr %2) nounwind ; <ptr> [#uses=1]
  store ptr %3, ptr %0, align 8
  %4 = load ptr, ptr %0, align 8                     ; <ptr> [#uses=1]
  store ptr %4, ptr %retval, align 8
  br label %return

return:                                           ; preds = %entry
  %retval1 = load ptr, ptr %retval                   ; <ptr> [#uses=1]
  ret ptr %retval1
}

define linkonce_odr ptr @_ZN5boost7details19compressed_pair_impI7empty_tiLi1EE5firstEv(ptr %this) nounwind ssp align 2 {
entry:
  %this_addr = alloca ptr, align 8 ; <ptr> [#uses=2]
  %retval = alloca ptr          ; <ptr> [#uses=2]
  %0 = alloca ptr               ; <ptr> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  store ptr %this, ptr %this_addr
  %1 = load ptr, ptr %this_addr, align 8 ; <ptr> [#uses=1]
  store ptr %1, ptr %0, align 8
  %2 = load ptr, ptr %0, align 8    ; <ptr> [#uses=1]
  store ptr %2, ptr %retval, align 8
  br label %return

return:                                           ; preds = %entry
  %retval1 = load ptr, ptr %retval  ; <ptr> [#uses=1]
  ret ptr %retval1
}

define linkonce_odr ptr @_ZN5boost15compressed_pairI7empty_tiE5firstEv(ptr %this) ssp align 2 {
entry:
  %this_addr = alloca ptr, align 8 ; <ptr> [#uses=2]
  %retval = alloca ptr          ; <ptr> [#uses=2]
  %0 = alloca ptr               ; <ptr> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  store ptr %this, ptr %this_addr
  %1 = load ptr, ptr %this_addr, align 8 ; <ptr> [#uses=1]
  %2 = getelementptr inbounds %"struct.boost::compressed_pair<empty_t,int>", ptr %1, i32 0, i32 0 ; <ptr> [#uses=1]
  %3 = call ptr @_ZN5boost7details19compressed_pair_impI7empty_tiLi1EE5firstEv(ptr %2) nounwind ; <ptr> [#uses=1]
  store ptr %3, ptr %0, align 8
  %4 = load ptr, ptr %0, align 8    ; <ptr> [#uses=1]
  store ptr %4, ptr %retval, align 8
  br label %return

return:                                           ; preds = %entry
  %retval1 = load ptr, ptr %retval  ; <ptr> [#uses=1]
  ret ptr %retval1
}

declare i32 @puts(ptr)

declare void @abort() noreturn
