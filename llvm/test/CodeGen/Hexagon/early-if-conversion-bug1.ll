; RUN: llc -O2 -mtriple=hexagon -mcpu=hexagonv5 < %s | FileCheck %s
; we do not want to see a segv.
; CHECK-NOT: segmentation
; CHECK: call

target datalayout = "e-m:e-p:32:32-i1:32-i64:64-a:0-v32:32-n16:32"
target triple = "hexagon"

%"class.std::__1::basic_string" = type { %"class.std::__1::__compressed_pair" }
%"class.std::__1::__compressed_pair" = type { %"class.std::__1::__libcpp_compressed_pair_imp" }
%"class.std::__1::__libcpp_compressed_pair_imp" = type { %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__rep" }
%"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__rep" = type { %union.anon }
%union.anon = type { %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__long" }
%"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__long" = type { i32, i32, ptr }
%"class.std::__1::ios_base" = type { ptr, i32, i32, i32, i32, i32, ptr, ptr, ptr, ptr, i32, i32, ptr, i32, i32, ptr, i32, i32 }
%"class.std::__1::basic_streambuf" = type { ptr, %"class.std::__1::locale", ptr, ptr, ptr, ptr, ptr, ptr }
%"class.std::__1::locale" = type { ptr }
%"class.std::__1::locale::__imp" = type opaque
%"class.std::__1::allocator" = type { i8 }
%"class.std::__1::ostreambuf_iterator" = type { ptr }
%"class.std::__1::__basic_string_common" = type { i8 }
%"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__short" = type { %union.anon.0, [11 x i8] }
%union.anon.0 = type { i8 }

; Function Attrs: nounwind
declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture readonly, i32, i1) #0

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nounwind
declare void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEED1Ev(ptr) #1

define weak_odr hidden i32 @_ZNSt3__116__pad_and_outputIcNS_11char_traitsIcEEEENS_19ostreambuf_iteratorIT_T0_EES6_PKS4_S8_S8_RNS_8ios_baseES4_(i32 %__s.coerce, ptr %__ob, ptr %__op, ptr %__oe, ptr nonnull %__iob, i8 zeroext %__fl) #2 personality ptr @__gxx_personality_v0 {
entry:
  %this.addr.i66 = alloca ptr, align 4
  %__s.addr.i67 = alloca ptr, align 4
  %__n.addr.i68 = alloca i32, align 4
  %__p.addr.i.i = alloca ptr, align 4
  %this.addr.i.i.i13.i.i = alloca ptr, align 4
  %this.addr.i.i14.i.i = alloca ptr, align 4
  %this.addr.i15.i.i = alloca ptr, align 4
  %__x.addr.i.i.i.i.i = alloca ptr, align 4
  %__r.addr.i.i.i.i = alloca ptr, align 4
  %this.addr.i.i.i4.i.i = alloca ptr, align 4
  %this.addr.i.i5.i.i = alloca ptr, align 4
  %this.addr.i6.i.i = alloca ptr, align 4
  %this.addr.i.i.i.i.i56 = alloca ptr, align 4
  %this.addr.i.i.i.i57 = alloca ptr, align 4
  %this.addr.i.i.i58 = alloca ptr, align 4
  %this.addr.i.i59 = alloca ptr, align 4
  %this.addr.i60 = alloca ptr, align 4
  %this.addr.i.i.i.i.i = alloca ptr, align 4
  %this.addr.i.i.i.i = alloca ptr, align 4
  %this.addr.i.i.i = alloca ptr, align 4
  %this.addr.i.i = alloca ptr, align 4
  %__n.addr.i.i = alloca i32, align 4
  %__c.addr.i.i = alloca i8, align 1
  %this.addr.i53 = alloca ptr, align 4
  %__n.addr.i54 = alloca i32, align 4
  %__c.addr.i = alloca i8, align 1
  %this.addr.i46 = alloca ptr, align 4
  %__s.addr.i47 = alloca ptr, align 4
  %__n.addr.i48 = alloca i32, align 4
  %this.addr.i44 = alloca ptr, align 4
  %__s.addr.i = alloca ptr, align 4
  %__n.addr.i = alloca i32, align 4
  %this.addr.i41 = alloca ptr, align 4
  %__wide.addr.i = alloca i32, align 4
  %__r.i = alloca i32, align 4
  %this.addr.i = alloca ptr, align 4
  %retval = alloca %"class.std::__1::ostreambuf_iterator", align 4
  %__s = alloca %"class.std::__1::ostreambuf_iterator", align 4
  %__ob.addr = alloca ptr, align 4
  %__op.addr = alloca ptr, align 4
  %__oe.addr = alloca ptr, align 4
  %__iob.addr = alloca ptr, align 4
  %__fl.addr = alloca i8, align 1
  %__sz = alloca i32, align 4
  %__ns = alloca i32, align 4
  %__np = alloca i32, align 4
  %__sp = alloca %"class.std::__1::basic_string", align 4
  %exn.slot = alloca ptr
  %ehselector.slot = alloca i32
  %cleanup.dest.slot = alloca i32
  %coerce.val.ip = inttoptr i32 %__s.coerce to ptr
  store ptr %coerce.val.ip, ptr %__s
  store ptr %__ob, ptr %__ob.addr, align 4
  store ptr %__op, ptr %__op.addr, align 4
  store ptr %__oe, ptr %__oe.addr, align 4
  store ptr %__iob, ptr %__iob.addr, align 4
  store i8 %__fl, ptr %__fl.addr, align 1
  %0 = load ptr, ptr %__s, align 4
  %cmp = icmp eq ptr %0, null
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %retval, ptr align 4 %__s, i32 4, i1 false)
  br label %return

if.end:                                           ; preds = %entry
  %1 = load ptr, ptr %__oe.addr, align 4
  %2 = load ptr, ptr %__ob.addr, align 4
  %sub.ptr.lhs.cast = ptrtoint ptr %1 to i32
  %sub.ptr.rhs.cast = ptrtoint ptr %2 to i32
  %sub.ptr.sub = sub i32 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  store i32 %sub.ptr.sub, ptr %__sz, align 4
  %3 = load ptr, ptr %__iob.addr, align 4
  store ptr %3, ptr %this.addr.i, align 4
  %this1.i = load ptr, ptr %this.addr.i
  %__width_.i = getelementptr inbounds %"class.std::__1::ios_base", ptr %this1.i, i32 0, i32 3
  %4 = load i32, ptr %__width_.i, align 4
  store i32 %4, ptr %__ns, align 4
  %5 = load i32, ptr %__ns, align 4
  %6 = load i32, ptr %__sz, align 4
  %cmp1 = icmp sgt i32 %5, %6
  br i1 %cmp1, label %if.then2, label %if.else

if.then2:                                         ; preds = %if.end
  %7 = load i32, ptr %__sz, align 4
  %8 = load i32, ptr %__ns, align 4
  %sub = sub nsw i32 %8, %7
  store i32 %sub, ptr %__ns, align 4
  br label %if.end3

if.else:                                          ; preds = %if.end
  store i32 0, ptr %__ns, align 4
  br label %if.end3

if.end3:                                          ; preds = %if.else, %if.then2
  %9 = load ptr, ptr %__op.addr, align 4
  %10 = load ptr, ptr %__ob.addr, align 4
  %sub.ptr.lhs.cast4 = ptrtoint ptr %9 to i32
  %sub.ptr.rhs.cast5 = ptrtoint ptr %10 to i32
  %sub.ptr.sub6 = sub i32 %sub.ptr.lhs.cast4, %sub.ptr.rhs.cast5
  store i32 %sub.ptr.sub6, ptr %__np, align 4
  %11 = load i32, ptr %__np, align 4
  %cmp7 = icmp sgt i32 %11, 0
  br i1 %cmp7, label %if.then8, label %if.end15

if.then8:                                         ; preds = %if.end3
  %12 = load ptr, ptr %__s, align 4
  %13 = load ptr, ptr %__ob.addr, align 4
  %14 = load i32, ptr %__np, align 4
  store ptr %12, ptr %this.addr.i46, align 4
  store ptr %13, ptr %__s.addr.i47, align 4
  store i32 %14, ptr %__n.addr.i48, align 4
  %this1.i49 = load ptr, ptr %this.addr.i46
  %vtable.i50 = load ptr, ptr %this1.i49
  %vfn.i51 = getelementptr inbounds ptr, ptr %vtable.i50, i64 12
  %15 = load ptr, ptr %vfn.i51
  %16 = load ptr, ptr %__s.addr.i47, align 4
  %17 = load i32, ptr %__n.addr.i48, align 4
  %call.i52 = call i32 %15(ptr %this1.i49, ptr %16, i32 %17)
  %18 = load i32, ptr %__np, align 4
  %cmp11 = icmp ne i32 %call.i52, %18
  br i1 %cmp11, label %if.then12, label %if.end14

if.then12:                                        ; preds = %if.then8
  store ptr null, ptr %__s, align 4
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %retval, ptr align 4 %__s, i32 4, i1 false)
  br label %return

if.end14:                                         ; preds = %if.then8
  br label %if.end15

if.end15:                                         ; preds = %if.end14, %if.end3
  %19 = load i32, ptr %__ns, align 4
  %cmp16 = icmp sgt i32 %19, 0
  br i1 %cmp16, label %if.then17, label %if.end25

if.then17:                                        ; preds = %if.end15
  %20 = load i32, ptr %__ns, align 4
  %21 = load i8, ptr %__fl.addr, align 1
  store ptr %__sp, ptr %this.addr.i53, align 4
  store i32 %20, ptr %__n.addr.i54, align 4
  store i8 %21, ptr %__c.addr.i, align 1
  %this1.i55 = load ptr, ptr %this.addr.i53
  %22 = load i32, ptr %__n.addr.i54, align 4
  %23 = load i8, ptr %__c.addr.i, align 1
  store ptr %this1.i55, ptr %this.addr.i.i, align 4
  store i32 %22, ptr %__n.addr.i.i, align 4
  store i8 %23, ptr %__c.addr.i.i, align 1
  %this1.i.i = load ptr, ptr %this.addr.i.i
  store ptr %this1.i.i, ptr %this.addr.i.i.i, align 4
  %this1.i.i.i = load ptr, ptr %this.addr.i.i.i
  store ptr %this1.i.i.i, ptr %this.addr.i.i.i.i, align 4
  %this1.i.i.i.i = load ptr, ptr %this.addr.i.i.i.i
  store ptr %this1.i.i.i.i, ptr %this.addr.i.i.i.i.i, align 4
  %this1.i.i.i.i.i = load ptr, ptr %this.addr.i.i.i.i.i
  %24 = load i32, ptr %__n.addr.i.i, align 4
  %25 = load i8, ptr %__c.addr.i.i, align 1
  call void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE6__initEjc(ptr %this1.i.i, i32 %24, i8 zeroext %25)
  %26 = load ptr, ptr %__s, align 4
  store ptr %__sp, ptr %this.addr.i60, align 4
  %this1.i61 = load ptr, ptr %this.addr.i60
  store ptr %this1.i61, ptr %this.addr.i.i59, align 4
  %this1.i.i62 = load ptr, ptr %this.addr.i.i59
  store ptr %this1.i.i62, ptr %this.addr.i.i.i58, align 4
  %this1.i.i.i63 = load ptr, ptr %this.addr.i.i.i58
  store ptr %this1.i.i.i63, ptr %this.addr.i.i.i.i57, align 4
  %this1.i.i.i.i64 = load ptr, ptr %this.addr.i.i.i.i57
  store ptr %this1.i.i.i.i64, ptr %this.addr.i.i.i.i.i56, align 4
  %this1.i.i.i.i.i65 = load ptr, ptr %this.addr.i.i.i.i.i56
  %27 = getelementptr inbounds %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__rep", ptr %this1.i.i.i.i.i65, i32 0, i32 0
  %28 = getelementptr inbounds %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__short", ptr %27, i32 0, i32 0
  %29 = load i8, ptr %28, align 1
  %conv.i.i.i = zext i8 %29 to i32
  %and.i.i.i = and i32 %conv.i.i.i, 1
  %tobool.i.i.i = icmp ne i32 %and.i.i.i, 0
  br i1 %tobool.i.i.i, label %cond.true.i.i, label %cond.false.i.i

cond.true.i.i:                                    ; preds = %if.then17
  store ptr %this1.i.i62, ptr %this.addr.i15.i.i, align 4
  %this1.i16.i.i = load ptr, ptr %this.addr.i15.i.i
  store ptr %this1.i16.i.i, ptr %this.addr.i.i14.i.i, align 4
  %this1.i.i18.i.i = load ptr, ptr %this.addr.i.i14.i.i
  store ptr %this1.i.i18.i.i, ptr %this.addr.i.i.i13.i.i, align 4
  %this1.i.i.i19.i.i = load ptr, ptr %this.addr.i.i.i13.i.i
  %30 = getelementptr inbounds %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__rep", ptr %this1.i.i.i19.i.i, i32 0, i32 0
  %__data_.i21.i.i = getelementptr inbounds %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__long", ptr %30, i32 0, i32 2
  %31 = load ptr, ptr %__data_.i21.i.i, align 4
  br label %_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE4dataEv.exit

cond.false.i.i:                                   ; preds = %if.then17
  store ptr %this1.i.i62, ptr %this.addr.i6.i.i, align 4
  %this1.i7.i.i = load ptr, ptr %this.addr.i6.i.i
  store ptr %this1.i7.i.i, ptr %this.addr.i.i5.i.i, align 4
  %this1.i.i9.i.i = load ptr, ptr %this.addr.i.i5.i.i
  store ptr %this1.i.i9.i.i, ptr %this.addr.i.i.i4.i.i, align 4
  %this1.i.i.i10.i.i = load ptr, ptr %this.addr.i.i.i4.i.i
  %32 = getelementptr inbounds %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__rep", ptr %this1.i.i.i10.i.i, i32 0, i32 0
  %__data_.i.i.i = getelementptr inbounds %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__short", ptr %32, i32 0, i32 1
  store ptr %__data_.i.i.i, ptr %__r.addr.i.i.i.i, align 4
  %33 = load ptr, ptr %__r.addr.i.i.i.i, align 4
  store ptr %33, ptr %__x.addr.i.i.i.i.i, align 4
  %34 = load ptr, ptr %__x.addr.i.i.i.i.i, align 4
  br label %_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE4dataEv.exit

_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE4dataEv.exit: ; preds = %cond.false.i.i, %cond.true.i.i
  %cond.i.i = phi ptr [ %31, %cond.true.i.i ], [ %34, %cond.false.i.i ]
  store ptr %cond.i.i, ptr %__p.addr.i.i, align 4
  %35 = load ptr, ptr %__p.addr.i.i, align 4
  %36 = load i32, ptr %__ns, align 4
  store ptr %26, ptr %this.addr.i66, align 4
  store ptr %35, ptr %__s.addr.i67, align 4
  store i32 %36, ptr %__n.addr.i68, align 4
  %this1.i69 = load ptr, ptr %this.addr.i66
  %vtable.i70 = load ptr, ptr %this1.i69
  %vfn.i71 = getelementptr inbounds ptr, ptr %vtable.i70, i64 12
  %37 = load ptr, ptr %vfn.i71
  %38 = load ptr, ptr %__s.addr.i67, align 4
  %39 = load i32, ptr %__n.addr.i68, align 4
  %call.i7273 = invoke i32 %37(ptr %this1.i69, ptr %38, i32 %39)
          to label %_ZNSt3__115basic_streambufIcNS_11char_traitsIcEEE5sputnEPKci.exit unwind label %lpad

_ZNSt3__115basic_streambufIcNS_11char_traitsIcEEE5sputnEPKci.exit: ; preds = %_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE4dataEv.exit
  br label %invoke.cont

invoke.cont:                                      ; preds = %_ZNSt3__115basic_streambufIcNS_11char_traitsIcEEE5sputnEPKci.exit
  %40 = load i32, ptr %__ns, align 4
  %cmp21 = icmp ne i32 %call.i7273, %40
  br i1 %cmp21, label %if.then22, label %if.end24

if.then22:                                        ; preds = %invoke.cont
  store ptr null, ptr %__s, align 4
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %retval, ptr align 4 %__s, i32 4, i1 false)
  store i32 1, ptr %cleanup.dest.slot
  br label %cleanup

lpad:                                             ; preds = %_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE4dataEv.exit
  %41 = landingpad { ptr, i32 }
          cleanup
  %42 = extractvalue { ptr, i32 } %41, 0
  store ptr %42, ptr %exn.slot
  %43 = extractvalue { ptr, i32 } %41, 1
  store i32 %43, ptr %ehselector.slot
  call void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEED1Ev(ptr %__sp) #0
  br label %eh.resume

if.end24:                                         ; preds = %invoke.cont
  store i32 0, ptr %cleanup.dest.slot
  br label %cleanup

cleanup:                                          ; preds = %if.end24, %if.then22
  call void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEED1Ev(ptr %__sp) #0
  %cleanup.dest = load i32, ptr %cleanup.dest.slot
  switch i32 %cleanup.dest, label %unreachable [
    i32 0, label %cleanup.cont
    i32 1, label %return
  ]

cleanup.cont:                                     ; preds = %cleanup
  br label %if.end25

if.end25:                                         ; preds = %cleanup.cont, %if.end15
  %44 = load ptr, ptr %__oe.addr, align 4
  %45 = load ptr, ptr %__op.addr, align 4
  %sub.ptr.lhs.cast26 = ptrtoint ptr %44 to i32
  %sub.ptr.rhs.cast27 = ptrtoint ptr %45 to i32
  %sub.ptr.sub28 = sub i32 %sub.ptr.lhs.cast26, %sub.ptr.rhs.cast27
  store i32 %sub.ptr.sub28, ptr %__np, align 4
  %46 = load i32, ptr %__np, align 4
  %cmp29 = icmp sgt i32 %46, 0
  br i1 %cmp29, label %if.then30, label %if.end37

if.then30:                                        ; preds = %if.end25
  %47 = load ptr, ptr %__s, align 4
  %48 = load ptr, ptr %__op.addr, align 4
  %49 = load i32, ptr %__np, align 4
  store ptr %47, ptr %this.addr.i44, align 4
  store ptr %48, ptr %__s.addr.i, align 4
  store i32 %49, ptr %__n.addr.i, align 4
  %this1.i45 = load ptr, ptr %this.addr.i44
  %vtable.i = load ptr, ptr %this1.i45
  %vfn.i = getelementptr inbounds ptr, ptr %vtable.i, i64 12
  %50 = load ptr, ptr %vfn.i
  %51 = load ptr, ptr %__s.addr.i, align 4
  %52 = load i32, ptr %__n.addr.i, align 4
  %call.i = call i32 %50(ptr %this1.i45, ptr %51, i32 %52)
  %53 = load i32, ptr %__np, align 4
  %cmp33 = icmp ne i32 %call.i, %53
  br i1 %cmp33, label %if.then34, label %if.end36

if.then34:                                        ; preds = %if.then30
  store ptr null, ptr %__s, align 4
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %retval, ptr align 4 %__s, i32 4, i1 false)
  br label %return

if.end36:                                         ; preds = %if.then30
  br label %if.end37

if.end37:                                         ; preds = %if.end36, %if.end25
  %54 = load ptr, ptr %__iob.addr, align 4
  store ptr %54, ptr %this.addr.i41, align 4
  store i32 0, ptr %__wide.addr.i, align 4
  %this1.i42 = load ptr, ptr %this.addr.i41
  %__width_.i43 = getelementptr inbounds %"class.std::__1::ios_base", ptr %this1.i42, i32 0, i32 3
  %55 = load i32, ptr %__width_.i43, align 4
  store i32 %55, ptr %__r.i, align 4
  %56 = load i32, ptr %__wide.addr.i, align 4
  %__width_2.i = getelementptr inbounds %"class.std::__1::ios_base", ptr %this1.i42, i32 0, i32 3
  store i32 %56, ptr %__width_2.i, align 4
  %57 = load i32, ptr %__r.i, align 4
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %retval, ptr align 4 %__s, i32 4, i1 false)
  br label %return

return:                                           ; preds = %if.end37, %if.then34, %cleanup, %if.then12, %if.then
  %58 = load ptr, ptr %retval
  %coerce.val.pi = ptrtoint ptr %58 to i32
  ret i32 %coerce.val.pi

eh.resume:                                        ; preds = %lpad
  %exn = load ptr, ptr %exn.slot
  %sel = load i32, ptr %ehselector.slot
  %lpad.val = insertvalue { ptr, i32 } undef, ptr %exn, 0
  %lpad.val40 = insertvalue { ptr, i32 } %lpad.val, i32 %sel, 1
  resume { ptr, i32 } %lpad.val40

unreachable:                                      ; preds = %cleanup
  unreachable
}

declare void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE6__initEjc(ptr, i32, i8 zeroext) #2

attributes #0 = { nounwind }
attributes #1 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"Clang 3.1"}
