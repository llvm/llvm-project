; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_EXT_image_raw10_raw12 %s -o - | FileCheck %s 
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s --spirv-ext=+SPV_EXT_image_raw10_raw12 -o - -filetype=obj | spirv-val %}

; CHECK: OpExtension "SPV_EXT_image_raw10_raw12"

  define dso_local spir_kernel void @test_raw1012(ptr addrspace(1) noundef writeonly align 4 captures(none) %dst, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %img) {
  entry:
    %call = tail call spir_func i32 @_Z27get_image_channel_data_type14ocl_image2d_ro(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %img)
    switch i32 %call, label %sw.epilog [
      i32 4304, label %sw.epilog.sink.split
      i32 4323, label %sw.bb1
      i32 4324, label %sw.bb2
    ]

  sw.bb1:                                           
    br label %sw.epilog.sink.split

  sw.bb2:                                           
    br label %sw.epilog.sink.split

  sw.epilog.sink.split:                             
    %.sink = phi i32 [ 12, %sw.bb2 ], [ 10, %sw.bb1 ], [ 8, %entry ]
    store i32 %.sink, ptr addrspace(1) %dst, align 4
    br label %sw.epilog

  sw.epilog:                                       
    %call3 = tail call spir_func i32 @_Z27get_image_channel_data_type14ocl_image2d_ro(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %img)
    %cmp = icmp eq i32 %call3, 4323
    br i1 %cmp, label %if.end7.sink.split, label %if.else

  if.else:                                          
    %call4 = tail call spir_func i32 @_Z27get_image_channel_data_type14ocl_image2d_ro(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %img)
    %cmp5 = icmp eq i32 %call4, 4324
    br i1 %cmp5, label %if.end7.sink.split, label %if.end7

  if.end7.sink.split:                               
    %.sink14 = phi i32 [ 1010, %sw.epilog ], [ 1212, %if.else ]
    store i32 %.sink14, ptr addrspace(1) %dst, align 4
    br label %if.end7

  if.end7:                                         
    ret void
  }

  declare spir_func i32 @_Z27get_image_channel_data_type14ocl_image2d_ro(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0))
