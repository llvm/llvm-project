; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_EXT_image_raw10_raw12 %s -o - | FileCheck %s 
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s --spirv-ext=+SPV_EXT_image_raw10_raw12 -o - -filetype=obj | spirv-val %}

; CHECK-NOT: OpExtension "SPV_EXT_image_raw10_raw12"

define dso_local spir_kernel void @test_raw1012(ptr addrspace(1) noundef writeonly align 4 captures(none) %dst, i32 noundef %value) {
entry:
  switch i32 %value, label %sw.epilog [
    i32 4323, label %sw.epilog.sink.split
    i32 4324, label %sw.bb1
  ]

sw.bb1:                                           
  br label %sw.epilog.sink.split

sw.epilog.sink.split:                             
  %.sink = phi i32 [ 12, %sw.bb1 ], [ 10, %entry ]
  store i32 %.sink, ptr addrspace(1) %dst, align 4
  br label %sw.epilog

sw.epilog:                                        
  %0 = add i32 %value, -4323
  %or.cond = icmp ult i32 %0, 2
  br i1 %or.cond, label %if.then, label %if.end

if.then:                                          
  store i32 1012, ptr addrspace(1) %dst, align 4
  br label %if.end

if.end:                                          
  ret void
}
