; RUN: llc < %s -mtriple=thumbv8-eabi -float-abi=hard | FileCheck %s
; RUN: llc < %s -mtriple=thumbv7-eabi -float-abi=hard -arm-restrict-it | FileCheck %s
; RUN: llc < %s -mtriple=thumbv8-eabi -float-abi=hard -regalloc=basic | FileCheck %s
; RUN: llc < %s -mtriple=thumbv7-eabi -float-abi=hard -regalloc=basic -arm-restrict-it | FileCheck %s

%"struct.__gnu_cxx::__normal_iterator<char*,std::basic_string<char, std::char_traits<char>, std::allocator<char> > >" = type { i8* }
%"struct.__gnu_cxx::new_allocator<char>" = type <{ i8 }>
%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >" = type { %"struct.__gnu_cxx::__normal_iterator<char*,std::basic_string<char, std::char_traits<char>, std::allocator<char> > >" }
%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >::_Rep" = type { %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >::_Rep_base" }
%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >::_Rep_base" = type { i32, i32, i32 }


define weak arm_aapcs_vfpcc i32 @_ZNKSs7compareERKSs(%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"* %this, %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"* %__str) {
; CHECK-LABEL: _ZNKSs7compareERKSs:
; CHECK:      cbz	r0,
; CHECK-NEXT: %bb1
; CHECK-NEXT: pop.w
; CHECK-NEXT: %bb
; CHECK-NEXT: sub{{(.w)?}} r0, r{{[0-9]+}}, r{{[0-9]+}}
; CHECK-NEXT: pop.w
entry:
  %0 = tail call arm_aapcs_vfpcc  i32 @_ZNKSs4sizeEv(%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"* %this) ; <i32> [#uses=3]
  %1 = tail call arm_aapcs_vfpcc  i32 @_ZNKSs4sizeEv(%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"* %__str) ; <i32> [#uses=3]
  %2 = icmp ult i32 %1, %0                        ; <i1> [#uses=1]
  %3 = select i1 %2, i32 %1, i32 %0               ; <i32> [#uses=1]
  %4 = tail call arm_aapcs_vfpcc  i8* @_ZNKSs7_M_dataEv(%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"* %this) ; <i8*> [#uses=1]
  %5 = tail call arm_aapcs_vfpcc  i8* @_ZNKSs4dataEv(%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"* %__str) ; <i8*> [#uses=1]
  %6 = tail call arm_aapcs_vfpcc  i32 @memcmp(i8* %4, i8* %5, i32 %3) nounwind readonly ; <i32> [#uses=2]
  %7 = icmp eq i32 %6, 0                          ; <i1> [#uses=1]
  br i1 %7, label %bb, label %bb1

bb:                                               ; preds = %entry
  %8 = sub i32 %0, %1                             ; <i32> [#uses=1]
  ret i32 %8

bb1:                                              ; preds = %entry
  ret i32 %6
}

declare arm_aapcs_vfpcc i32 @memcmp(i8* nocapture, i8* nocapture, i32) nounwind readonly

declare arm_aapcs_vfpcc i32 @_ZNKSs4sizeEv(%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"* %this)

declare arm_aapcs_vfpcc i8* @_ZNKSs7_M_dataEv(%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"* %this)

declare arm_aapcs_vfpcc i8* @_ZNKSs4dataEv(%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"* %this)
