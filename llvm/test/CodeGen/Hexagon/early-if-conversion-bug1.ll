; RUN: llc -O2 -march=hexagon -mcpu=hexagonv5 < %s | FileCheck %s
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
%"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__long" = type { i32, i32, i8* }
%"class.std::__1::ios_base" = type { i32 (...)**, i32, i32, i32, i32, i32, i8*, i8*, void (i8, %"class.std::__1::ios_base"*, i32)**, i32*, i32, i32, i32*, i32, i32, i8**, i32, i32 }
%"class.std::__1::basic_streambuf" = type { i32 (...)**, %"class.std::__1::locale", i8*, i8*, i8*, i8*, i8*, i8* }
%"class.std::__1::locale" = type { %"class.std::__1::locale::__imp"* }
%"class.std::__1::locale::__imp" = type opaque
%"class.std::__1::allocator" = type { i8 }
%"class.std::__1::ostreambuf_iterator" = type { %"class.std::__1::basic_streambuf"* }
%"class.std::__1::__basic_string_common" = type { i8 }
%"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__short" = type { %union.anon.0, [11 x i8] }
%union.anon.0 = type { i8 }

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture readonly, i32, i1) #0

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nounwind
declare void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEED1Ev(%"class.std::__1::basic_string"*) #1

define weak_odr hidden i32 @_ZNSt3__116__pad_and_outputIcNS_11char_traitsIcEEEENS_19ostreambuf_iteratorIT_T0_EES6_PKS4_S8_S8_RNS_8ios_baseES4_(i32 %__s.coerce, i8* %__ob, i8* %__op, i8* %__oe, %"class.std::__1::ios_base"* nonnull %__iob, i8 zeroext %__fl) #2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %this.addr.i66 = alloca %"class.std::__1::basic_streambuf"*, align 4
  %__s.addr.i67 = alloca i8*, align 4
  %__n.addr.i68 = alloca i32, align 4
  %__p.addr.i.i = alloca i8*, align 4
  %this.addr.i.i.i13.i.i = alloca %"class.std::__1::__libcpp_compressed_pair_imp"*, align 4
  %this.addr.i.i14.i.i = alloca %"class.std::__1::__compressed_pair"*, align 4
  %this.addr.i15.i.i = alloca %"class.std::__1::basic_string"*, align 4
  %__x.addr.i.i.i.i.i = alloca i8*, align 4
  %__r.addr.i.i.i.i = alloca i8*, align 4
  %this.addr.i.i.i4.i.i = alloca %"class.std::__1::__libcpp_compressed_pair_imp"*, align 4
  %this.addr.i.i5.i.i = alloca %"class.std::__1::__compressed_pair"*, align 4
  %this.addr.i6.i.i = alloca %"class.std::__1::basic_string"*, align 4
  %this.addr.i.i.i.i.i56 = alloca %"class.std::__1::__libcpp_compressed_pair_imp"*, align 4
  %this.addr.i.i.i.i57 = alloca %"class.std::__1::__compressed_pair"*, align 4
  %this.addr.i.i.i58 = alloca %"class.std::__1::basic_string"*, align 4
  %this.addr.i.i59 = alloca %"class.std::__1::basic_string"*, align 4
  %this.addr.i60 = alloca %"class.std::__1::basic_string"*, align 4
  %this.addr.i.i.i.i.i = alloca %"class.std::__1::allocator"*, align 4
  %this.addr.i.i.i.i = alloca %"class.std::__1::__libcpp_compressed_pair_imp"*, align 4
  %this.addr.i.i.i = alloca %"class.std::__1::__compressed_pair"*, align 4
  %this.addr.i.i = alloca %"class.std::__1::basic_string"*, align 4
  %__n.addr.i.i = alloca i32, align 4
  %__c.addr.i.i = alloca i8, align 1
  %this.addr.i53 = alloca %"class.std::__1::basic_string"*, align 4
  %__n.addr.i54 = alloca i32, align 4
  %__c.addr.i = alloca i8, align 1
  %this.addr.i46 = alloca %"class.std::__1::basic_streambuf"*, align 4
  %__s.addr.i47 = alloca i8*, align 4
  %__n.addr.i48 = alloca i32, align 4
  %this.addr.i44 = alloca %"class.std::__1::basic_streambuf"*, align 4
  %__s.addr.i = alloca i8*, align 4
  %__n.addr.i = alloca i32, align 4
  %this.addr.i41 = alloca %"class.std::__1::ios_base"*, align 4
  %__wide.addr.i = alloca i32, align 4
  %__r.i = alloca i32, align 4
  %this.addr.i = alloca %"class.std::__1::ios_base"*, align 4
  %retval = alloca %"class.std::__1::ostreambuf_iterator", align 4
  %__s = alloca %"class.std::__1::ostreambuf_iterator", align 4
  %__ob.addr = alloca i8*, align 4
  %__op.addr = alloca i8*, align 4
  %__oe.addr = alloca i8*, align 4
  %__iob.addr = alloca %"class.std::__1::ios_base"*, align 4
  %__fl.addr = alloca i8, align 1
  %__sz = alloca i32, align 4
  %__ns = alloca i32, align 4
  %__np = alloca i32, align 4
  %__sp = alloca %"class.std::__1::basic_string", align 4
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  %cleanup.dest.slot = alloca i32
  %coerce.dive = getelementptr %"class.std::__1::ostreambuf_iterator", %"class.std::__1::ostreambuf_iterator"* %__s, i32 0, i32 0
  %coerce.val.ip = inttoptr i32 %__s.coerce to %"class.std::__1::basic_streambuf"*
  store %"class.std::__1::basic_streambuf"* %coerce.val.ip, %"class.std::__1::basic_streambuf"** %coerce.dive
  store i8* %__ob, i8** %__ob.addr, align 4
  store i8* %__op, i8** %__op.addr, align 4
  store i8* %__oe, i8** %__oe.addr, align 4
  store %"class.std::__1::ios_base"* %__iob, %"class.std::__1::ios_base"** %__iob.addr, align 4
  store i8 %__fl, i8* %__fl.addr, align 1
  %__sbuf_ = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", %"class.std::__1::ostreambuf_iterator"* %__s, i32 0, i32 0
  %0 = load %"class.std::__1::basic_streambuf"*, %"class.std::__1::basic_streambuf"** %__sbuf_, align 4
  %cmp = icmp eq %"class.std::__1::basic_streambuf"* %0, null
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %1 = bitcast %"class.std::__1::ostreambuf_iterator"* %retval to i8*
  %2 = bitcast %"class.std::__1::ostreambuf_iterator"* %__s to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %1, i8* align 4 %2, i32 4, i1 false)
  br label %return

if.end:                                           ; preds = %entry
  %3 = load i8*, i8** %__oe.addr, align 4
  %4 = load i8*, i8** %__ob.addr, align 4
  %sub.ptr.lhs.cast = ptrtoint i8* %3 to i32
  %sub.ptr.rhs.cast = ptrtoint i8* %4 to i32
  %sub.ptr.sub = sub i32 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  store i32 %sub.ptr.sub, i32* %__sz, align 4
  %5 = load %"class.std::__1::ios_base"*, %"class.std::__1::ios_base"** %__iob.addr, align 4
  store %"class.std::__1::ios_base"* %5, %"class.std::__1::ios_base"** %this.addr.i, align 4
  %this1.i = load %"class.std::__1::ios_base"*, %"class.std::__1::ios_base"** %this.addr.i
  %__width_.i = getelementptr inbounds %"class.std::__1::ios_base", %"class.std::__1::ios_base"* %this1.i, i32 0, i32 3
  %6 = load i32, i32* %__width_.i, align 4
  store i32 %6, i32* %__ns, align 4
  %7 = load i32, i32* %__ns, align 4
  %8 = load i32, i32* %__sz, align 4
  %cmp1 = icmp sgt i32 %7, %8
  br i1 %cmp1, label %if.then2, label %if.else

if.then2:                                         ; preds = %if.end
  %9 = load i32, i32* %__sz, align 4
  %10 = load i32, i32* %__ns, align 4
  %sub = sub nsw i32 %10, %9
  store i32 %sub, i32* %__ns, align 4
  br label %if.end3

if.else:                                          ; preds = %if.end
  store i32 0, i32* %__ns, align 4
  br label %if.end3

if.end3:                                          ; preds = %if.else, %if.then2
  %11 = load i8*, i8** %__op.addr, align 4
  %12 = load i8*, i8** %__ob.addr, align 4
  %sub.ptr.lhs.cast4 = ptrtoint i8* %11 to i32
  %sub.ptr.rhs.cast5 = ptrtoint i8* %12 to i32
  %sub.ptr.sub6 = sub i32 %sub.ptr.lhs.cast4, %sub.ptr.rhs.cast5
  store i32 %sub.ptr.sub6, i32* %__np, align 4
  %13 = load i32, i32* %__np, align 4
  %cmp7 = icmp sgt i32 %13, 0
  br i1 %cmp7, label %if.then8, label %if.end15

if.then8:                                         ; preds = %if.end3
  %__sbuf_9 = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", %"class.std::__1::ostreambuf_iterator"* %__s, i32 0, i32 0
  %14 = load %"class.std::__1::basic_streambuf"*, %"class.std::__1::basic_streambuf"** %__sbuf_9, align 4
  %15 = load i8*, i8** %__ob.addr, align 4
  %16 = load i32, i32* %__np, align 4
  store %"class.std::__1::basic_streambuf"* %14, %"class.std::__1::basic_streambuf"** %this.addr.i46, align 4
  store i8* %15, i8** %__s.addr.i47, align 4
  store i32 %16, i32* %__n.addr.i48, align 4
  %this1.i49 = load %"class.std::__1::basic_streambuf"*, %"class.std::__1::basic_streambuf"** %this.addr.i46
  %17 = bitcast %"class.std::__1::basic_streambuf"* %this1.i49 to i32 (%"class.std::__1::basic_streambuf"*, i8*, i32)***
  %vtable.i50 = load i32 (%"class.std::__1::basic_streambuf"*, i8*, i32)**, i32 (%"class.std::__1::basic_streambuf"*, i8*, i32)*** %17
  %vfn.i51 = getelementptr inbounds i32 (%"class.std::__1::basic_streambuf"*, i8*, i32)*, i32 (%"class.std::__1::basic_streambuf"*, i8*, i32)** %vtable.i50, i64 12
  %18 = load i32 (%"class.std::__1::basic_streambuf"*, i8*, i32)*, i32 (%"class.std::__1::basic_streambuf"*, i8*, i32)** %vfn.i51
  %19 = load i8*, i8** %__s.addr.i47, align 4
  %20 = load i32, i32* %__n.addr.i48, align 4
  %call.i52 = call i32 %18(%"class.std::__1::basic_streambuf"* %this1.i49, i8* %19, i32 %20)
  %21 = load i32, i32* %__np, align 4
  %cmp11 = icmp ne i32 %call.i52, %21
  br i1 %cmp11, label %if.then12, label %if.end14

if.then12:                                        ; preds = %if.then8
  %__sbuf_13 = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", %"class.std::__1::ostreambuf_iterator"* %__s, i32 0, i32 0
  store %"class.std::__1::basic_streambuf"* null, %"class.std::__1::basic_streambuf"** %__sbuf_13, align 4
  %22 = bitcast %"class.std::__1::ostreambuf_iterator"* %retval to i8*
  %23 = bitcast %"class.std::__1::ostreambuf_iterator"* %__s to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %22, i8* align 4 %23, i32 4, i1 false)
  br label %return

if.end14:                                         ; preds = %if.then8
  br label %if.end15

if.end15:                                         ; preds = %if.end14, %if.end3
  %24 = load i32, i32* %__ns, align 4
  %cmp16 = icmp sgt i32 %24, 0
  br i1 %cmp16, label %if.then17, label %if.end25

if.then17:                                        ; preds = %if.end15
  %25 = load i32, i32* %__ns, align 4
  %26 = load i8, i8* %__fl.addr, align 1
  store %"class.std::__1::basic_string"* %__sp, %"class.std::__1::basic_string"** %this.addr.i53, align 4
  store i32 %25, i32* %__n.addr.i54, align 4
  store i8 %26, i8* %__c.addr.i, align 1
  %this1.i55 = load %"class.std::__1::basic_string"*, %"class.std::__1::basic_string"** %this.addr.i53
  %27 = load i32, i32* %__n.addr.i54, align 4
  %28 = load i8, i8* %__c.addr.i, align 1
  store %"class.std::__1::basic_string"* %this1.i55, %"class.std::__1::basic_string"** %this.addr.i.i, align 4
  store i32 %27, i32* %__n.addr.i.i, align 4
  store i8 %28, i8* %__c.addr.i.i, align 1
  %this1.i.i = load %"class.std::__1::basic_string"*, %"class.std::__1::basic_string"** %this.addr.i.i
  %29 = bitcast %"class.std::__1::basic_string"* %this1.i.i to %"class.std::__1::__basic_string_common"*
  %__r_.i.i = getelementptr inbounds %"class.std::__1::basic_string", %"class.std::__1::basic_string"* %this1.i.i, i32 0, i32 0
  store %"class.std::__1::__compressed_pair"* %__r_.i.i, %"class.std::__1::__compressed_pair"** %this.addr.i.i.i, align 4
  %this1.i.i.i = load %"class.std::__1::__compressed_pair"*, %"class.std::__1::__compressed_pair"** %this.addr.i.i.i
  %30 = bitcast %"class.std::__1::__compressed_pair"* %this1.i.i.i to %"class.std::__1::__libcpp_compressed_pair_imp"*
  store %"class.std::__1::__libcpp_compressed_pair_imp"* %30, %"class.std::__1::__libcpp_compressed_pair_imp"** %this.addr.i.i.i.i, align 4
  %this1.i.i.i.i = load %"class.std::__1::__libcpp_compressed_pair_imp"*, %"class.std::__1::__libcpp_compressed_pair_imp"** %this.addr.i.i.i.i
  %31 = bitcast %"class.std::__1::__libcpp_compressed_pair_imp"* %this1.i.i.i.i to %"class.std::__1::allocator"*
  store %"class.std::__1::allocator"* %31, %"class.std::__1::allocator"** %this.addr.i.i.i.i.i, align 4
  %this1.i.i.i.i.i = load %"class.std::__1::allocator"*, %"class.std::__1::allocator"** %this.addr.i.i.i.i.i
  %__first_.i.i.i.i = getelementptr inbounds %"class.std::__1::__libcpp_compressed_pair_imp", %"class.std::__1::__libcpp_compressed_pair_imp"* %this1.i.i.i.i, i32 0, i32 0
  %32 = load i32, i32* %__n.addr.i.i, align 4
  %33 = load i8, i8* %__c.addr.i.i, align 1
  call void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE6__initEjc(%"class.std::__1::basic_string"* %this1.i.i, i32 %32, i8 zeroext %33)
  %__sbuf_18 = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", %"class.std::__1::ostreambuf_iterator"* %__s, i32 0, i32 0
  %34 = load %"class.std::__1::basic_streambuf"*, %"class.std::__1::basic_streambuf"** %__sbuf_18, align 4
  store %"class.std::__1::basic_string"* %__sp, %"class.std::__1::basic_string"** %this.addr.i60, align 4
  %this1.i61 = load %"class.std::__1::basic_string"*, %"class.std::__1::basic_string"** %this.addr.i60
  store %"class.std::__1::basic_string"* %this1.i61, %"class.std::__1::basic_string"** %this.addr.i.i59, align 4
  %this1.i.i62 = load %"class.std::__1::basic_string"*, %"class.std::__1::basic_string"** %this.addr.i.i59
  store %"class.std::__1::basic_string"* %this1.i.i62, %"class.std::__1::basic_string"** %this.addr.i.i.i58, align 4
  %this1.i.i.i63 = load %"class.std::__1::basic_string"*, %"class.std::__1::basic_string"** %this.addr.i.i.i58
  %__r_.i.i.i = getelementptr inbounds %"class.std::__1::basic_string", %"class.std::__1::basic_string"* %this1.i.i.i63, i32 0, i32 0
  store %"class.std::__1::__compressed_pair"* %__r_.i.i.i, %"class.std::__1::__compressed_pair"** %this.addr.i.i.i.i57, align 4
  %this1.i.i.i.i64 = load %"class.std::__1::__compressed_pair"*, %"class.std::__1::__compressed_pair"** %this.addr.i.i.i.i57
  %35 = bitcast %"class.std::__1::__compressed_pair"* %this1.i.i.i.i64 to %"class.std::__1::__libcpp_compressed_pair_imp"*
  store %"class.std::__1::__libcpp_compressed_pair_imp"* %35, %"class.std::__1::__libcpp_compressed_pair_imp"** %this.addr.i.i.i.i.i56, align 4
  %this1.i.i.i.i.i65 = load %"class.std::__1::__libcpp_compressed_pair_imp"*, %"class.std::__1::__libcpp_compressed_pair_imp"** %this.addr.i.i.i.i.i56
  %__first_.i.i.i.i.i = getelementptr inbounds %"class.std::__1::__libcpp_compressed_pair_imp", %"class.std::__1::__libcpp_compressed_pair_imp"* %this1.i.i.i.i.i65, i32 0, i32 0
  %36 = getelementptr inbounds %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__rep", %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__rep"* %__first_.i.i.i.i.i, i32 0, i32 0
  %__s.i.i.i = bitcast %union.anon* %36 to %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__short"*
  %37 = getelementptr inbounds %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__short", %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__short"* %__s.i.i.i, i32 0, i32 0
  %__size_.i.i.i = bitcast %union.anon.0* %37 to i8*
  %38 = load i8, i8* %__size_.i.i.i, align 1
  %conv.i.i.i = zext i8 %38 to i32
  %and.i.i.i = and i32 %conv.i.i.i, 1
  %tobool.i.i.i = icmp ne i32 %and.i.i.i, 0
  br i1 %tobool.i.i.i, label %cond.true.i.i, label %cond.false.i.i

cond.true.i.i:                                    ; preds = %if.then17
  store %"class.std::__1::basic_string"* %this1.i.i62, %"class.std::__1::basic_string"** %this.addr.i15.i.i, align 4
  %this1.i16.i.i = load %"class.std::__1::basic_string"*, %"class.std::__1::basic_string"** %this.addr.i15.i.i
  %__r_.i17.i.i = getelementptr inbounds %"class.std::__1::basic_string", %"class.std::__1::basic_string"* %this1.i16.i.i, i32 0, i32 0
  store %"class.std::__1::__compressed_pair"* %__r_.i17.i.i, %"class.std::__1::__compressed_pair"** %this.addr.i.i14.i.i, align 4
  %this1.i.i18.i.i = load %"class.std::__1::__compressed_pair"*, %"class.std::__1::__compressed_pair"** %this.addr.i.i14.i.i
  %39 = bitcast %"class.std::__1::__compressed_pair"* %this1.i.i18.i.i to %"class.std::__1::__libcpp_compressed_pair_imp"*
  store %"class.std::__1::__libcpp_compressed_pair_imp"* %39, %"class.std::__1::__libcpp_compressed_pair_imp"** %this.addr.i.i.i13.i.i, align 4
  %this1.i.i.i19.i.i = load %"class.std::__1::__libcpp_compressed_pair_imp"*, %"class.std::__1::__libcpp_compressed_pair_imp"** %this.addr.i.i.i13.i.i
  %__first_.i.i.i20.i.i = getelementptr inbounds %"class.std::__1::__libcpp_compressed_pair_imp", %"class.std::__1::__libcpp_compressed_pair_imp"* %this1.i.i.i19.i.i, i32 0, i32 0
  %40 = getelementptr inbounds %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__rep", %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__rep"* %__first_.i.i.i20.i.i, i32 0, i32 0
  %__l.i.i.i = bitcast %union.anon* %40 to %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__long"*
  %__data_.i21.i.i = getelementptr inbounds %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__long", %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__long"* %__l.i.i.i, i32 0, i32 2
  %41 = load i8*, i8** %__data_.i21.i.i, align 4
  br label %_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE4dataEv.exit

cond.false.i.i:                                   ; preds = %if.then17
  store %"class.std::__1::basic_string"* %this1.i.i62, %"class.std::__1::basic_string"** %this.addr.i6.i.i, align 4
  %this1.i7.i.i = load %"class.std::__1::basic_string"*, %"class.std::__1::basic_string"** %this.addr.i6.i.i
  %__r_.i8.i.i = getelementptr inbounds %"class.std::__1::basic_string", %"class.std::__1::basic_string"* %this1.i7.i.i, i32 0, i32 0
  store %"class.std::__1::__compressed_pair"* %__r_.i8.i.i, %"class.std::__1::__compressed_pair"** %this.addr.i.i5.i.i, align 4
  %this1.i.i9.i.i = load %"class.std::__1::__compressed_pair"*, %"class.std::__1::__compressed_pair"** %this.addr.i.i5.i.i
  %42 = bitcast %"class.std::__1::__compressed_pair"* %this1.i.i9.i.i to %"class.std::__1::__libcpp_compressed_pair_imp"*
  store %"class.std::__1::__libcpp_compressed_pair_imp"* %42, %"class.std::__1::__libcpp_compressed_pair_imp"** %this.addr.i.i.i4.i.i, align 4
  %this1.i.i.i10.i.i = load %"class.std::__1::__libcpp_compressed_pair_imp"*, %"class.std::__1::__libcpp_compressed_pair_imp"** %this.addr.i.i.i4.i.i
  %__first_.i.i.i11.i.i = getelementptr inbounds %"class.std::__1::__libcpp_compressed_pair_imp", %"class.std::__1::__libcpp_compressed_pair_imp"* %this1.i.i.i10.i.i, i32 0, i32 0
  %43 = getelementptr inbounds %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__rep", %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__rep"* %__first_.i.i.i11.i.i, i32 0, i32 0
  %__s.i12.i.i = bitcast %union.anon* %43 to %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__short"*
  %__data_.i.i.i = getelementptr inbounds %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__short", %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__short"* %__s.i12.i.i, i32 0, i32 1
  %arrayidx.i.i.i = getelementptr inbounds [11 x i8], [11 x i8]* %__data_.i.i.i, i32 0, i32 0
  store i8* %arrayidx.i.i.i, i8** %__r.addr.i.i.i.i, align 4
  %44 = load i8*, i8** %__r.addr.i.i.i.i, align 4
  store i8* %44, i8** %__x.addr.i.i.i.i.i, align 4
  %45 = load i8*, i8** %__x.addr.i.i.i.i.i, align 4
  br label %_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE4dataEv.exit

_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE4dataEv.exit: ; preds = %cond.false.i.i, %cond.true.i.i
  %cond.i.i = phi i8* [ %41, %cond.true.i.i ], [ %45, %cond.false.i.i ]
  store i8* %cond.i.i, i8** %__p.addr.i.i, align 4
  %46 = load i8*, i8** %__p.addr.i.i, align 4
  %47 = load i32, i32* %__ns, align 4
  store %"class.std::__1::basic_streambuf"* %34, %"class.std::__1::basic_streambuf"** %this.addr.i66, align 4
  store i8* %46, i8** %__s.addr.i67, align 4
  store i32 %47, i32* %__n.addr.i68, align 4
  %this1.i69 = load %"class.std::__1::basic_streambuf"*, %"class.std::__1::basic_streambuf"** %this.addr.i66
  %48 = bitcast %"class.std::__1::basic_streambuf"* %this1.i69 to i32 (%"class.std::__1::basic_streambuf"*, i8*, i32)***
  %vtable.i70 = load i32 (%"class.std::__1::basic_streambuf"*, i8*, i32)**, i32 (%"class.std::__1::basic_streambuf"*, i8*, i32)*** %48
  %vfn.i71 = getelementptr inbounds i32 (%"class.std::__1::basic_streambuf"*, i8*, i32)*, i32 (%"class.std::__1::basic_streambuf"*, i8*, i32)** %vtable.i70, i64 12
  %49 = load i32 (%"class.std::__1::basic_streambuf"*, i8*, i32)*, i32 (%"class.std::__1::basic_streambuf"*, i8*, i32)** %vfn.i71
  %50 = load i8*, i8** %__s.addr.i67, align 4
  %51 = load i32, i32* %__n.addr.i68, align 4
  %call.i7273 = invoke i32 %49(%"class.std::__1::basic_streambuf"* %this1.i69, i8* %50, i32 %51)
          to label %_ZNSt3__115basic_streambufIcNS_11char_traitsIcEEE5sputnEPKci.exit unwind label %lpad

_ZNSt3__115basic_streambufIcNS_11char_traitsIcEEE5sputnEPKci.exit: ; preds = %_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE4dataEv.exit
  br label %invoke.cont

invoke.cont:                                      ; preds = %_ZNSt3__115basic_streambufIcNS_11char_traitsIcEEE5sputnEPKci.exit
  %52 = load i32, i32* %__ns, align 4
  %cmp21 = icmp ne i32 %call.i7273, %52
  br i1 %cmp21, label %if.then22, label %if.end24

if.then22:                                        ; preds = %invoke.cont
  %__sbuf_23 = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", %"class.std::__1::ostreambuf_iterator"* %__s, i32 0, i32 0
  store %"class.std::__1::basic_streambuf"* null, %"class.std::__1::basic_streambuf"** %__sbuf_23, align 4
  %53 = bitcast %"class.std::__1::ostreambuf_iterator"* %retval to i8*
  %54 = bitcast %"class.std::__1::ostreambuf_iterator"* %__s to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %53, i8* align 4 %54, i32 4, i1 false)
  store i32 1, i32* %cleanup.dest.slot
  br label %cleanup

lpad:                                             ; preds = %_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE4dataEv.exit
  %55 = landingpad { i8*, i32 }
          cleanup
  %56 = extractvalue { i8*, i32 } %55, 0
  store i8* %56, i8** %exn.slot
  %57 = extractvalue { i8*, i32 } %55, 1
  store i32 %57, i32* %ehselector.slot
  call void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEED1Ev(%"class.std::__1::basic_string"* %__sp) #0
  br label %eh.resume

if.end24:                                         ; preds = %invoke.cont
  store i32 0, i32* %cleanup.dest.slot
  br label %cleanup

cleanup:                                          ; preds = %if.end24, %if.then22
  call void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEED1Ev(%"class.std::__1::basic_string"* %__sp) #0
  %cleanup.dest = load i32, i32* %cleanup.dest.slot
  switch i32 %cleanup.dest, label %unreachable [
    i32 0, label %cleanup.cont
    i32 1, label %return
  ]

cleanup.cont:                                     ; preds = %cleanup
  br label %if.end25

if.end25:                                         ; preds = %cleanup.cont, %if.end15
  %58 = load i8*, i8** %__oe.addr, align 4
  %59 = load i8*, i8** %__op.addr, align 4
  %sub.ptr.lhs.cast26 = ptrtoint i8* %58 to i32
  %sub.ptr.rhs.cast27 = ptrtoint i8* %59 to i32
  %sub.ptr.sub28 = sub i32 %sub.ptr.lhs.cast26, %sub.ptr.rhs.cast27
  store i32 %sub.ptr.sub28, i32* %__np, align 4
  %60 = load i32, i32* %__np, align 4
  %cmp29 = icmp sgt i32 %60, 0
  br i1 %cmp29, label %if.then30, label %if.end37

if.then30:                                        ; preds = %if.end25
  %__sbuf_31 = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", %"class.std::__1::ostreambuf_iterator"* %__s, i32 0, i32 0
  %61 = load %"class.std::__1::basic_streambuf"*, %"class.std::__1::basic_streambuf"** %__sbuf_31, align 4
  %62 = load i8*, i8** %__op.addr, align 4
  %63 = load i32, i32* %__np, align 4
  store %"class.std::__1::basic_streambuf"* %61, %"class.std::__1::basic_streambuf"** %this.addr.i44, align 4
  store i8* %62, i8** %__s.addr.i, align 4
  store i32 %63, i32* %__n.addr.i, align 4
  %this1.i45 = load %"class.std::__1::basic_streambuf"*, %"class.std::__1::basic_streambuf"** %this.addr.i44
  %64 = bitcast %"class.std::__1::basic_streambuf"* %this1.i45 to i32 (%"class.std::__1::basic_streambuf"*, i8*, i32)***
  %vtable.i = load i32 (%"class.std::__1::basic_streambuf"*, i8*, i32)**, i32 (%"class.std::__1::basic_streambuf"*, i8*, i32)*** %64
  %vfn.i = getelementptr inbounds i32 (%"class.std::__1::basic_streambuf"*, i8*, i32)*, i32 (%"class.std::__1::basic_streambuf"*, i8*, i32)** %vtable.i, i64 12
  %65 = load i32 (%"class.std::__1::basic_streambuf"*, i8*, i32)*, i32 (%"class.std::__1::basic_streambuf"*, i8*, i32)** %vfn.i
  %66 = load i8*, i8** %__s.addr.i, align 4
  %67 = load i32, i32* %__n.addr.i, align 4
  %call.i = call i32 %65(%"class.std::__1::basic_streambuf"* %this1.i45, i8* %66, i32 %67)
  %68 = load i32, i32* %__np, align 4
  %cmp33 = icmp ne i32 %call.i, %68
  br i1 %cmp33, label %if.then34, label %if.end36

if.then34:                                        ; preds = %if.then30
  %__sbuf_35 = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", %"class.std::__1::ostreambuf_iterator"* %__s, i32 0, i32 0
  store %"class.std::__1::basic_streambuf"* null, %"class.std::__1::basic_streambuf"** %__sbuf_35, align 4
  %69 = bitcast %"class.std::__1::ostreambuf_iterator"* %retval to i8*
  %70 = bitcast %"class.std::__1::ostreambuf_iterator"* %__s to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %69, i8* align 4 %70, i32 4, i1 false)
  br label %return

if.end36:                                         ; preds = %if.then30
  br label %if.end37

if.end37:                                         ; preds = %if.end36, %if.end25
  %71 = load %"class.std::__1::ios_base"*, %"class.std::__1::ios_base"** %__iob.addr, align 4
  store %"class.std::__1::ios_base"* %71, %"class.std::__1::ios_base"** %this.addr.i41, align 4
  store i32 0, i32* %__wide.addr.i, align 4
  %this1.i42 = load %"class.std::__1::ios_base"*, %"class.std::__1::ios_base"** %this.addr.i41
  %__width_.i43 = getelementptr inbounds %"class.std::__1::ios_base", %"class.std::__1::ios_base"* %this1.i42, i32 0, i32 3
  %72 = load i32, i32* %__width_.i43, align 4
  store i32 %72, i32* %__r.i, align 4
  %73 = load i32, i32* %__wide.addr.i, align 4
  %__width_2.i = getelementptr inbounds %"class.std::__1::ios_base", %"class.std::__1::ios_base"* %this1.i42, i32 0, i32 3
  store i32 %73, i32* %__width_2.i, align 4
  %74 = load i32, i32* %__r.i, align 4
  %75 = bitcast %"class.std::__1::ostreambuf_iterator"* %retval to i8*
  %76 = bitcast %"class.std::__1::ostreambuf_iterator"* %__s to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %75, i8* align 4 %76, i32 4, i1 false)
  br label %return

return:                                           ; preds = %if.end37, %if.then34, %cleanup, %if.then12, %if.then
  %coerce.dive39 = getelementptr %"class.std::__1::ostreambuf_iterator", %"class.std::__1::ostreambuf_iterator"* %retval, i32 0, i32 0
  %77 = load %"class.std::__1::basic_streambuf"*, %"class.std::__1::basic_streambuf"** %coerce.dive39
  %coerce.val.pi = ptrtoint %"class.std::__1::basic_streambuf"* %77 to i32
  ret i32 %coerce.val.pi

eh.resume:                                        ; preds = %lpad
  %exn = load i8*, i8** %exn.slot
  %sel = load i32, i32* %ehselector.slot
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn, 0
  %lpad.val40 = insertvalue { i8*, i32 } %lpad.val, i32 %sel, 1
  resume { i8*, i32 } %lpad.val40

unreachable:                                      ; preds = %cleanup
  unreachable
}

declare void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE6__initEjc(%"class.std::__1::basic_string"*, i32, i8 zeroext) #2

attributes #0 = { nounwind }
attributes #1 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"Clang 3.1"}
