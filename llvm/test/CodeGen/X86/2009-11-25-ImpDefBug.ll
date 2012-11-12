; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu
; pr5600

%struct..0__pthread_mutex_s = type { i32, i32, i32, i32, i32, i32, %struct.__pthread_list_t }
%struct.ASN1ObjHeader = type { i8, %"struct.__gmp_expr<__mpz_struct [1],__mpz_struct [1]>", i64, i32, i32, i32 }
%struct.ASN1Object = type { i32 (...)**, i32, i32, i64 }
%struct.ASN1Unit = type { [4 x i32 (%struct.ASN1ObjHeader*, %struct.ASN1Object**)*], %"struct.std::ASN1ObjList" }
%"struct.__gmp_expr<__mpz_struct [1],__mpz_struct [1]>" = type { [1 x %struct.__mpz_struct] }
%struct.__mpz_struct = type { i32, i32, i64* }
%struct.__pthread_list_t = type { %struct.__pthread_list_t*, %struct.__pthread_list_t* }
%struct.pthread_attr_t = type { i64, [48 x i8] }
%struct.pthread_mutex_t = type { %struct..0__pthread_mutex_s }
%struct.pthread_mutexattr_t = type { i32 }
%"struct.std::ASN1ObjList" = type { %"struct.std::_Vector_base<ASN1Object*,std::allocator<ASN1Object*> >" }
%"struct.std::_Vector_base<ASN1Object*,std::allocator<ASN1Object*> >" = type { %"struct.std::_Vector_base<ASN1Object*,std::allocator<ASN1Object*> >::_Vector_impl" }
%"struct.std::_Vector_base<ASN1Object*,std::allocator<ASN1Object*> >::_Vector_impl" = type { %struct.ASN1Object**, %struct.ASN1Object**, %struct.ASN1Object** }
%struct.xmstream = type { i8*, i64, i64, i64, i8 }

declare void @_ZNSt6vectorIP10ASN1ObjectSaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_(%"struct.std::ASN1ObjList"* nocapture, i64, %struct.ASN1Object** nocapture)

declare i32 @_Z17LoadObjectFromBERR8xmstreamPP10ASN1ObjectPPF10ASN1StatusP13ASN1ObjHeaderS3_E(%struct.xmstream*, %struct.ASN1Object**, i32 (%struct.ASN1ObjHeader*, %struct.ASN1Object**)**)

define i32 @_ZN8ASN1Unit4loadER8xmstreamjm18ASN1LengthEncoding(%struct.ASN1Unit* %this, %struct.xmstream* nocapture %stream, i32 %numObjects, i64 %size, i32 %lEncoding) {
entry:
  br label %meshBB85

bb5:                                              ; preds = %bb13.fragment.cl135, %bb13.fragment.cl, %bb.i.i.bbcl.disp, %bb13.fragment
  %0 = invoke i32 @_Z17LoadObjectFromBERR8xmstreamPP10ASN1ObjectPPF10ASN1StatusP13ASN1ObjHeaderS3_E(%struct.xmstream* undef, %struct.ASN1Object** undef, i32 (%struct.ASN1ObjHeader*, %struct.ASN1Object**)** undef)
          to label %meshBB81.bbcl.disp unwind label %lpad ; <i32> [#uses=0]

bb10.fragment:                                    ; preds = %bb13.fragment.bbcl.disp
  br i1 undef, label %bb1.i.fragment.bbcl.disp, label %bb.i.i.bbcl.disp

bb1.i.fragment:                                   ; preds = %bb1.i.fragment.bbcl.disp
  invoke void @_ZNSt6vectorIP10ASN1ObjectSaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_(%"struct.std::ASN1ObjList"* undef, i64 undef, %struct.ASN1Object** undef)
          to label %meshBB81.bbcl.disp unwind label %lpad

bb13.fragment:                                    ; preds = %bb13.fragment.bbcl.disp
  br i1 undef, label %meshBB81.bbcl.disp, label %bb5

bb.i4:                                            ; preds = %bb.i4.bbcl.disp, %bb1.i.fragment.bbcl.disp
  ret i32 undef

bb1.i5:                                           ; preds = %bb.i1
  ret i32 undef

lpad:                                             ; preds = %bb1.i.fragment.cl, %bb1.i.fragment, %bb5
  %.SV10.phi807 = phi i8* [ undef, %bb1.i.fragment.cl ], [ undef, %bb1.i.fragment ], [ undef, %bb5 ] ; <i8*> [#uses=1]
  %exn = landingpad {i8*, i32} personality i32 (...)* @__gxx_personality_v0
            cleanup
  %1 = load i8* %.SV10.phi807, align 8            ; <i8> [#uses=0]
  br i1 undef, label %meshBB81.bbcl.disp, label %bb13.fragment.bbcl.disp

bb.i1:                                            ; preds = %bb.i.i.bbcl.disp
  br i1 undef, label %meshBB81.bbcl.disp, label %bb1.i5

meshBB81:                                         ; preds = %meshBB81.bbcl.disp, %bb.i.i.bbcl.disp
  br i1 undef, label %meshBB81.bbcl.disp, label %bb.i4.bbcl.disp

meshBB85:                                         ; preds = %meshBB81.bbcl.disp, %bb.i4.bbcl.disp, %bb1.i.fragment.bbcl.disp, %bb.i.i.bbcl.disp, %entry
  br i1 undef, label %meshBB81.bbcl.disp, label %bb13.fragment.bbcl.disp

bb.i.i.bbcl.disp:                                 ; preds = %bb10.fragment
  switch i8 undef, label %meshBB85 [
    i8 123, label %bb.i1
    i8 97, label %bb5
    i8 44, label %meshBB81
    i8 1, label %meshBB81.cl
    i8 51, label %meshBB81.cl141
  ]

bb1.i.fragment.cl:                                ; preds = %bb1.i.fragment.bbcl.disp
  invoke void @_ZNSt6vectorIP10ASN1ObjectSaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_(%"struct.std::ASN1ObjList"* undef, i64 undef, %struct.ASN1Object** undef)
          to label %meshBB81.bbcl.disp unwind label %lpad

bb1.i.fragment.bbcl.disp:                         ; preds = %bb10.fragment
  switch i8 undef, label %bb.i4 [
    i8 97, label %bb1.i.fragment
    i8 7, label %bb1.i.fragment.cl
    i8 35, label %bb.i4.cl
    i8 77, label %meshBB85
  ]

bb13.fragment.cl:                                 ; preds = %bb13.fragment.bbcl.disp
  br i1 undef, label %meshBB81.bbcl.disp, label %bb5

bb13.fragment.cl135:                              ; preds = %bb13.fragment.bbcl.disp
  br i1 undef, label %meshBB81.bbcl.disp, label %bb5

bb13.fragment.bbcl.disp:                          ; preds = %meshBB85, %lpad
  switch i8 undef, label %bb10.fragment [
    i8 67, label %bb13.fragment.cl
    i8 108, label %bb13.fragment
    i8 58, label %bb13.fragment.cl135
  ]

bb.i4.cl:                                         ; preds = %bb.i4.bbcl.disp, %bb1.i.fragment.bbcl.disp
  ret i32 undef

bb.i4.bbcl.disp:                                  ; preds = %meshBB81.cl141, %meshBB81.cl, %meshBB81
  switch i8 undef, label %bb.i4 [
    i8 35, label %bb.i4.cl
    i8 77, label %meshBB85
  ]

meshBB81.cl:                                      ; preds = %meshBB81.bbcl.disp, %bb.i.i.bbcl.disp
  br i1 undef, label %meshBB81.bbcl.disp, label %bb.i4.bbcl.disp

meshBB81.cl141:                                   ; preds = %meshBB81.bbcl.disp, %bb.i.i.bbcl.disp
  br i1 undef, label %meshBB81.bbcl.disp, label %bb.i4.bbcl.disp

meshBB81.bbcl.disp:                               ; preds = %meshBB81.cl141, %meshBB81.cl, %bb13.fragment.cl135, %bb13.fragment.cl, %bb1.i.fragment.cl, %meshBB85, %meshBB81, %bb.i1, %lpad, %bb13.fragment, %bb1.i.fragment, %bb5
  switch i8 undef, label %meshBB85 [
    i8 44, label %meshBB81
    i8 1, label %meshBB81.cl
    i8 51, label %meshBB81.cl141
  ]
}

declare i32 @__gxx_personality_v0(...)
