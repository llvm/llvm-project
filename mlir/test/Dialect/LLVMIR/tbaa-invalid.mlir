// RUN: mlir-opt -split-input-file -verify-diagnostics %s

#tbaa_root = #ptr.tbaa_root<id = "Simple C/C++ TBAA">
#tbaa_desc = #ptr.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_tag = #ptr.tbaa_tag<access_type = #tbaa_desc, base_type = #tbaa_desc, offset = 0>
// expected-error@+2 {{invalid kind of attribute specified}}
// expected-error@below {{failed to parse Ptr_TBAATagAttr parameter 'access_type' which is to be a `TBAATypeDescriptorAttr`}}
#tbaa_tag2 = #ptr.tbaa_tag<access_type = #tbaa_tag, base_type = #tbaa_desc, offset = 0>

// -----

#tbaa_root = #ptr.tbaa_root<id = "Simple C/C++ TBAA">
#tbaa_desc = #ptr.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_tag = #ptr.tbaa_tag<access_type = #tbaa_desc, base_type = #tbaa_desc, offset = 0>
// expected-error@+2 {{invalid kind of attribute specified}}
// expected-error@below {{failed to parse Ptr_TBAATagAttr parameter 'base_type' which is to be a `TBAATypeDescriptorAttr`}}
#tbaa_tag2 = #ptr.tbaa_tag<access_type = #tbaa_desc, base_type = #tbaa_tag, offset = 0>

// -----

#tbaa_root = #ptr.tbaa_root<id = "Simple C/C++ TBAA">
#tbaa_desc = #ptr.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_tag = #ptr.tbaa_tag<access_type = #tbaa_desc, base_type = #tbaa_desc, offset = 0>
// expected-error@+3 {{invalid kind of attribute specified}}
// expected-error@+2 {{failed to parse Ptr_TBAAMemberAttr parameter 'typeDesc' which is to be a `TBAANodeAttr`}}
// expected-error@below {{failed to parse Ptr_TBAATypeDescriptorAttr parameter 'members' which is to be a `::llvm::ArrayRef<TBAAMemberAttr>`}}
#tbaa_desc2 = #ptr.tbaa_type_desc<id = "long long", members = {<#tbaa_tag, 0>}>
