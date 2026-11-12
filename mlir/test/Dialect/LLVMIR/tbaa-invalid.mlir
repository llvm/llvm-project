// RUN: mlir-opt -split-input-file -verify-diagnostics %s

#tbaa_root = #llvm.tbaa_root<id = "Simple C/C++ TBAA">
#tbaa_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<access_type = #tbaa_desc, base_type = #tbaa_desc, offset = 0>
// expected-error@+2 {{invalid kind of attribute specified}}
// expected-error@below {{failed to parse LLVM_TBAATagAttr parameter 'access_type' which is to be a `TBAATypeDescriptorAttr`}}
#tbaa_tag2 = #llvm.tbaa_tag<access_type = #tbaa_tag, base_type = #tbaa_desc, offset = 0>

// -----

#tbaa_root = #llvm.tbaa_root<id = "Simple C/C++ TBAA">
#tbaa_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<access_type = #tbaa_desc, base_type = #tbaa_desc, offset = 0>
// expected-error@+2 {{invalid kind of attribute specified}}
// expected-error@below {{failed to parse LLVM_TBAATagAttr parameter 'base_type' which is to be a `TBAATypeDescriptorAttr`}}
#tbaa_tag2 = #llvm.tbaa_tag<access_type = #tbaa_desc, base_type = #tbaa_tag, offset = 0>

// -----

#tbaa_root = #llvm.tbaa_root<id = "Simple C/C++ TBAA">
#tbaa_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<access_type = #tbaa_desc, base_type = #tbaa_desc, offset = 0>
// expected-error@+3 {{invalid kind of attribute specified}}
// expected-error@+2 {{failed to parse LLVM_TBAAMemberAttr parameter 'typeDesc' which is to be a `TBAANodeAttr`}}
// expected-error@below {{failed to parse LLVM_TBAATypeDescriptorAttr parameter 'members' which is to be a `::llvm::ArrayRef<TBAAMemberAttr>`}}
#tbaa_desc2 = #llvm.tbaa_type_desc<id = "long long", members = {<#tbaa_tag, 0>}>
