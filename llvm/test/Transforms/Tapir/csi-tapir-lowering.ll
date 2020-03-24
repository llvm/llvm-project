; RUN: opt < %s -loop-spawning-ti -simplifycfg -csi -csi-instrument-basic-blocks=false -csi-instrument-memory-accesses=false -csi-instrument-atomics=false -csi-instrument-memintrinsics=false -csi-instrument-allocfn=false -csi-instrument-alloca=false -csi-instrument-function-calls=false -S -o - | FileCheck %s
; RUN: opt < %s -passes='loop-spawning,function(simplify-cfg),csi' -csi-instrument-basic-blocks=false -csi-instrument-memory-accesses=false -csi-instrument-atomics=false -csi-instrument-memintrinsics=false -csi-instrument-allocfn=false -csi-instrument-alloca=false -csi-instrument-function-calls=false -S -o - | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.cv::Mat" = type { i32, i32, i32, i32, i8*, i8*, i8*, i8*, %"class.cv::MatAllocator"*, %"struct.cv::UMatData"*, %"struct.cv::MatSize", %"struct.cv::MatStep" }
%"class.cv::MatAllocator" = type { i32 (...)** }
%"struct.cv::UMatData" = type { %"class.cv::MatAllocator"*, %"class.cv::MatAllocator"*, i32, i32, i8*, i8*, i64, i32, i8*, i8*, i32, i32, %"struct.cv::UMatData"* }
%"struct.cv::MatSize" = type { i32* }
%"struct.cv::MatStep" = type { i64*, [2 x i64] }
%"class.cv::String" = type { i8*, i64 }
%"class.cv::_InputArray" = type { i32, i8*, %"class.cv::Size_" }
%"class.cv::Size_" = type { i32, i32 }
%"class.std::vector.95" = type { %"struct.std::_Vector_base.96" }
%"struct.std::_Vector_base.96" = type { %"struct.std::_Vector_base<int, std::allocator<int> >::_Vector_impl" }
%"struct.std::_Vector_base<int, std::allocator<int> >::_Vector_impl" = type { i32*, i32*, i32* }
%"class.tfk::Render" = type { i8 }
%"class.tfk::Stack" = type { i32, %"class.std::basic_string", %"class.std::basic_string", i32, i32, i8, i32, i32, i32, i32, i8, %"struct.std::pair", %"class.std::vector.129", %class.Graph*, [2 x %"class.tfk::MLBase"*], [2 x %"class.tfk::ParamDB"*], %struct._align_data* }
%"class.std::basic_string" = type { %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider" }
%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider" = type { i8* }
%"struct.std::pair" = type { %"class.cv::Point_", %"class.cv::Point_" }
%"class.cv::Point_" = type { float, float }
%"class.std::vector.129" = type { %"struct.std::_Vector_base.130" }
%"struct.std::_Vector_base.130" = type { %"struct.std::_Vector_base<tfk::Section *, std::allocator<tfk::Section *> >::_Vector_impl" }
%"struct.std::_Vector_base<tfk::Section *, std::allocator<tfk::Section *> >::_Vector_impl" = type { %"class.tfk::Section"**, %"class.tfk::Section"**, %"class.tfk::Section"** }
%"class.tfk::Section" = type { i32, i32, i32, i32, i32, i8, i32, i32, i8, %"struct.std::pair", %"struct.std::pair", %"class.cv::Mat"*, %"class.std::vector"*, %"class.std::basic_string", %"class.std::basic_string", %struct._align_data*, %"class.std::vector.3", %"class.std::vector.8", %class.Graph*, %"class.std::mutex"*, double, double, double, double, double, double, %"class.std::vector.38"*, %"class.std::vector.38"*, %"class.std::vector.38"*, %"class.cv::Point_"*, %"class.cv::Point_"*, double*, double*, %"class.std::set", %"class.std::vector.109", %"class.cv::Mat", %"class.cv::Mat", %"class.cv::Mat", i32, i32, %"class.tfk::MLBase"**, %"class.tfk::ParamDB"**, %"class.tfk::TriangleMesh"* }
%"class.std::vector" = type { %"struct.std::_Vector_base" }
%"struct.std::_Vector_base" = type { %"struct.std::_Vector_base<cv::KeyPoint, std::allocator<cv::KeyPoint> >::_Vector_impl" }
%"struct.std::_Vector_base<cv::KeyPoint, std::allocator<cv::KeyPoint> >::_Vector_impl" = type { %"class.cv::KeyPoint"*, %"class.cv::KeyPoint"*, %"class.cv::KeyPoint"* }
%"class.cv::KeyPoint" = type { %"class.cv::Point_", float, float, float, i32, i32 }
%"class.std::vector.3" = type { %"struct.std::_Vector_base.4" }
%"struct.std::_Vector_base.4" = type { %"struct.std::_Vector_base<cv::Mat, std::allocator<cv::Mat> >::_Vector_impl" }
%"struct.std::_Vector_base<cv::Mat, std::allocator<cv::Mat> >::_Vector_impl" = type { %"class.cv::Mat"*, %"class.cv::Mat"*, %"class.cv::Mat"* }
%"class.std::vector.8" = type { %"struct.std::_Vector_base.9" }
%"struct.std::_Vector_base.9" = type { %"struct.std::_Vector_base<tfk::Tile *, std::allocator<tfk::Tile *> >::_Vector_impl" }
%"struct.std::_Vector_base<tfk::Tile *, std::allocator<tfk::Tile *> >::_Vector_impl" = type { %"class.tfk::Tile"**, %"class.tfk::Tile"**, %"class.tfk::Tile"** }
%"class.tfk::Tile" = type <{ %"class.std::map", %"class.std::map.16", %"class.std::map.21", double, i32, i8, [3 x i8], i32, [4 x i8], %"class.std::map", %"class.std::map", i8, [7 x i8], %"class.std::map.26", i32, i32, i32, i32, %"class.std::basic_string", %"class.cv::Mat"*, double, double, double, double, double, double, double, double, i8, i8, [6 x i8], double, double, %"class.tfk::MatchTilesTask"*, %"class.std::vector"*, %"class.cv::Mat"*, %"class.std::vector"*, %"class.cv::Mat"*, %"class.std::vector"*, %"class.cv::Mat"*, %"class.std::vector"*, %"class.cv::Mat"*, double, double, double, double, double, double, i8*, %"class.std::vector.31"*, i32, i8, [3 x i8], %class.TileData, i8, i8, [6 x i8], %"class.cv::Mat", %"class.cv::Mat", %"class.std::mutex"*, %"class.std::mutex"*, i8, [7 x i8], %"class.std::vector.33", %"class.std::vector.33", %"class.tfk::MLBase"**, %"class.tfk::ParamDB"**, %"class.std::map.71", %"class.std::map.78", %"class.std::map.83", %"class.std::map.83", i32, [4 x i8] }>
%"class.std::map.16" = type { %"class.std::_Rb_tree.17" }
%"class.std::_Rb_tree.17" = type { %"struct.std::_Rb_tree<int, std::pair<const int, std::pair<cv::Point_<float>, cv::Point_<float> > >, std::_Select1st<std::pair<const int, std::pair<cv::Point_<float>, cv::Point_<float> > > >, std::less<int>, std::allocator<std::pair<const int, std::pair<cv::Point_<float>, cv::Point_<float> > > > >::_Rb_tree_impl" }
%"struct.std::_Rb_tree<int, std::pair<const int, std::pair<cv::Point_<float>, cv::Point_<float> > >, std::_Select1st<std::pair<const int, std::pair<cv::Point_<float>, cv::Point_<float> > > >, std::less<int>, std::allocator<std::pair<const int, std::pair<cv::Point_<float>, cv::Point_<float> > > > >::_Rb_tree_impl" = type { %"struct.std::less", %"struct.std::_Rb_tree_node_base", i64 }
%"struct.std::less" = type { i8 }
%"struct.std::_Rb_tree_node_base" = type { i32, %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"* }
%"class.std::map.21" = type { %"class.std::_Rb_tree.22" }
%"class.std::_Rb_tree.22" = type { %"struct.std::_Rb_tree<int, std::pair<const int, double>, std::_Select1st<std::pair<const int, double> >, std::less<int>, std::allocator<std::pair<const int, double> > >::_Rb_tree_impl" }
%"struct.std::_Rb_tree<int, std::pair<const int, double>, std::_Select1st<std::pair<const int, double> >, std::less<int>, std::allocator<std::pair<const int, double> > >::_Rb_tree_impl" = type { %"struct.std::less", %"struct.std::_Rb_tree_node_base", i64 }
%"class.std::map" = type { %"class.std::_Rb_tree" }
%"class.std::_Rb_tree" = type { %"struct.std::_Rb_tree<int, std::pair<const int, cv::Point_<float> >, std::_Select1st<std::pair<const int, cv::Point_<float> > >, std::less<int>, std::allocator<std::pair<const int, cv::Point_<float> > > >::_Rb_tree_impl" }
%"struct.std::_Rb_tree<int, std::pair<const int, cv::Point_<float> >, std::_Select1st<std::pair<const int, cv::Point_<float> > >, std::less<int>, std::allocator<std::pair<const int, cv::Point_<float> > > >::_Rb_tree_impl" = type { %"struct.std::less", %"struct.std::_Rb_tree_node_base", i64 }
%"class.std::map.26" = type { %"class.std::_Rb_tree.27" }
%"class.std::_Rb_tree.27" = type { %"struct.std::_Rb_tree<int, std::pair<const int, float>, std::_Select1st<std::pair<const int, float> >, std::less<int>, std::allocator<std::pair<const int, float> > >::_Rb_tree_impl" }
%"struct.std::_Rb_tree<int, std::pair<const int, float>, std::_Select1st<std::pair<const int, float> >, std::less<int>, std::allocator<std::pair<const int, float> > >::_Rb_tree_impl" = type { %"struct.std::less", %"struct.std::_Rb_tree_node_base", i64 }
%"class.tfk::MatchTilesTask" = type { %"class.tfk::MRTask.base", %"class.std::vector.95", %"class.std::vector.95", %"class.std::map.119", %"class.tfk::Tile"*, %"class.std::vector.8", %"class.std::vector.8", %"class.std::map.78", %"class.std::map.124" }
%"class.tfk::MRTask.base" = type <{ i32 (...)**, %"class.tfk::ParamDB"*, %"class.tfk::MRParams"*, %"class.tfk::MLBase"*, i32 }>
%"class.tfk::ParamDB" = type { i8, %"class.std::mutex"*, %struct._align_data*, %"class.std::vector.54" }
%"class.std::vector.54" = type { %"struct.std::_Vector_base.55" }
%"struct.std::_Vector_base.55" = type { %"struct.std::_Vector_base<tfk::MRParams *, std::allocator<tfk::MRParams *> >::_Vector_impl" }
%"struct.std::_Vector_base<tfk::MRParams *, std::allocator<tfk::MRParams *> >::_Vector_impl" = type { %"class.tfk::MRParams"**, %"class.tfk::MRParams"**, %"class.tfk::MRParams"** }
%"class.tfk::MRParams" = type { %"class.std::map.59", %"class.std::map.66", %"struct.tfk::stats", %"struct.tfk::stats", i32, i32, i32, %"class.std::mutex"* }
%"class.std::map.59" = type { %"class.std::_Rb_tree.60" }
%"class.std::_Rb_tree.60" = type { %"struct.std::_Rb_tree<std::basic_string<char>, std::pair<const std::basic_string<char>, float>, std::_Select1st<std::pair<const std::basic_string<char>, float> >, std::less<std::basic_string<char> >, std::allocator<std::pair<const std::basic_string<char>, float> > >::_Rb_tree_impl" }
%"struct.std::_Rb_tree<std::basic_string<char>, std::pair<const std::basic_string<char>, float>, std::_Select1st<std::pair<const std::basic_string<char>, float> >, std::less<std::basic_string<char> >, std::allocator<std::pair<const std::basic_string<char>, float> > >::_Rb_tree_impl" = type { %"struct.std::less.64", %"struct.std::_Rb_tree_node_base", i64 }
%"struct.std::less.64" = type { i8 }
%"class.std::map.66" = type { %"class.std::_Rb_tree.67" }
%"class.std::_Rb_tree.67" = type { %"struct.std::_Rb_tree<std::basic_string<char>, std::pair<const std::basic_string<char>, int>, std::_Select1st<std::pair<const std::basic_string<char>, int> >, std::less<std::basic_string<char> >, std::allocator<std::pair<const std::basic_string<char>, int> > >::_Rb_tree_impl" }
%"struct.std::_Rb_tree<std::basic_string<char>, std::pair<const std::basic_string<char>, int>, std::_Select1st<std::pair<const std::basic_string<char>, int> >, std::less<std::basic_string<char> >, std::allocator<std::pair<const std::basic_string<char>, int> > >::_Rb_tree_impl" = type { %"struct.std::less.64", %"struct.std::_Rb_tree_node_base", i64 }
%"struct.tfk::stats" = type { i32, double, double }
%"class.tfk::MLBase" = type <{ i32 (...)**, %"class.std::recursive_mutex"*, %"class.std::vector.43", %"class.std::vector.48", %"struct.cv::Ptr", %"class.std::vector.43", %"class.std::vector.48", %"class.std::vector.48", %"struct.cv::Ptr.53", i32, i8, [3 x i8], i32, i32, i32, i32, i8, [3 x i8], i32, i32, [4 x i8] }>
%"class.std::recursive_mutex" = type { %"class.std::__recursive_mutex_base" }
%"class.std::__recursive_mutex_base" = type { %union.pthread_mutex_t }
%union.pthread_mutex_t = type { %"struct.(anonymous union)::__pthread_mutex_s" }
%"struct.(anonymous union)::__pthread_mutex_s" = type { i32, i32, i32, i32, i32, i16, i16, %struct.__pthread_internal_list }
%struct.__pthread_internal_list = type { %struct.__pthread_internal_list*, %struct.__pthread_internal_list* }
%"struct.cv::Ptr" = type { %"struct.cv::detail::PtrOwner"*, %"class.cv::ml::RTrees"* }
%"struct.cv::detail::PtrOwner" = type <{ i32 (...)**, i32, [4 x i8] }>
%"class.cv::ml::RTrees" = type { %"class.cv::ml::DTrees" }
%"class.cv::ml::DTrees" = type { %"class.cv::ml::StatModel" }
%"class.cv::ml::StatModel" = type { %"class.cv::Algorithm" }
%"class.cv::Algorithm" = type { i32 (...)** }
%"class.std::vector.43" = type { %"struct.std::_Vector_base.44" }
%"struct.std::_Vector_base.44" = type { %"struct.std::_Vector_base<std::vector<float, std::allocator<float> >, std::allocator<std::vector<float, std::allocator<float> > > >::_Vector_impl" }
%"struct.std::_Vector_base<std::vector<float, std::allocator<float> >, std::allocator<std::vector<float, std::allocator<float> > > >::_Vector_impl" = type { %"class.std::vector.48"*, %"class.std::vector.48"*, %"class.std::vector.48"* }
%"class.std::vector.48" = type { %"struct.std::_Vector_base.49" }
%"struct.std::_Vector_base.49" = type { %"struct.std::_Vector_base<float, std::allocator<float> >::_Vector_impl" }
%"struct.std::_Vector_base<float, std::allocator<float> >::_Vector_impl" = type { float*, float*, float* }
%"struct.cv::Ptr.53" = type { %"struct.cv::detail::PtrOwner"*, %"class.cv::ml::StatModel"* }
%"class.std::map.119" = type { %"class.std::_Rb_tree.120" }
%"class.std::_Rb_tree.120" = type { %"struct.std::_Rb_tree<int, std::pair<const int, tfk::TileSiftTask *>, std::_Select1st<std::pair<const int, tfk::TileSiftTask *> >, std::less<int>, std::allocator<std::pair<const int, tfk::TileSiftTask *> > >::_Rb_tree_impl" }
%"struct.std::_Rb_tree<int, std::pair<const int, tfk::TileSiftTask *>, std::_Select1st<std::pair<const int, tfk::TileSiftTask *> >, std::less<int>, std::allocator<std::pair<const int, tfk::TileSiftTask *> > >::_Rb_tree_impl" = type { %"struct.std::less", %"struct.std::_Rb_tree_node_base", i64 }
%"class.std::map.124" = type { %"class.std::_Rb_tree.125" }
%"class.std::_Rb_tree.125" = type { %"struct.std::_Rb_tree<tfk::Tile *, std::pair<tfk::Tile *const, tfk::MRTask *>, std::_Select1st<std::pair<tfk::Tile *const, tfk::MRTask *> >, std::less<tfk::Tile *>, std::allocator<std::pair<tfk::Tile *const, tfk::MRTask *> > >::_Rb_tree_impl" }
%"struct.std::_Rb_tree<tfk::Tile *, std::pair<tfk::Tile *const, tfk::MRTask *>, std::_Select1st<std::pair<tfk::Tile *const, tfk::MRTask *> >, std::less<tfk::Tile *>, std::allocator<std::pair<tfk::Tile *const, tfk::MRTask *> > >::_Rb_tree_impl" = type { %"struct.std::less.76", %"struct.std::_Rb_tree_node_base", i64 }
%"struct.std::less.76" = type { i8 }
%"class.std::vector.31" = type opaque
%class.TileData = type <{ %"class.google::protobuf::Message", %"class.google::protobuf::internal::InternalMetadataWithArena", %"class.google::protobuf::internal::HasBits", i32, %"class.google::protobuf::RepeatedPtrField", %"class.google::protobuf::RepeatedPtrField", %"class.google::protobuf::RepeatedPtrField.32", %"struct.google::protobuf::internal::ArenaStringPtr", %class.Matrix*, %class.Matrix*, %class.Matrix*, i64, i64, i64, i64, i64, i64, i64, i64, double, double, double, double, double, double, i64, i64, i64, i8, i8, [6 x i8] }>
%"class.google::protobuf::Message" = type { %"class.google::protobuf::MessageLite" }
%"class.google::protobuf::MessageLite" = type { i32 (...)** }
%"class.google::protobuf::internal::InternalMetadataWithArena" = type { %"class.google::protobuf::internal::InternalMetadataWithArenaBase" }
%"class.google::protobuf::internal::InternalMetadataWithArenaBase" = type { i8* }
%"class.google::protobuf::internal::HasBits" = type { [1 x i32] }
%"class.google::protobuf::RepeatedPtrField" = type { %"class.google::protobuf::internal::RepeatedPtrFieldBase" }
%"class.google::protobuf::internal::RepeatedPtrFieldBase" = type { %"class.google::protobuf::Arena"*, i32, i32, %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* }
%"class.google::protobuf::Arena" = type { %"class.google::protobuf::internal::ArenaImpl", i8* (%"class.google::protobuf::Arena"*)*, void (%"class.std::type_info"*, i64, i8*)*, void (%"class.google::protobuf::Arena"*, i8*, i64)*, void (%"class.google::protobuf::Arena"*, i8*, i64)*, i8* }
%"class.google::protobuf::internal::ArenaImpl" = type { i64, i64, i64, i8, %"class.google::protobuf::internal::Mutex", i64, %"struct.google::protobuf::internal::ArenaImpl::Options" }
%"class.google::protobuf::internal::Mutex" = type { %"struct.google::protobuf::internal::Mutex::Internal"* }
%"struct.google::protobuf::internal::Mutex::Internal" = type opaque
%"struct.google::protobuf::internal::ArenaImpl::Options" = type { i64, i64, i8*, i64, i8* (i64)*, void (i8*, i64)* }
%"class.std::type_info" = type { i32 (...)**, i8* }
%"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep" = type { i32, [1 x i8*] }
%"class.google::protobuf::RepeatedPtrField.32" = type { %"class.google::protobuf::internal::RepeatedPtrFieldBase" }
%"struct.google::protobuf::internal::ArenaStringPtr" = type { %"class.std::basic_string"* }
%class.Matrix = type { %"class.google::protobuf::Message", %"class.google::protobuf::internal::InternalMetadataWithArena", %"class.google::protobuf::internal::HasBits", i32, %"class.google::protobuf::RepeatedField", i64, i64 }
%"class.google::protobuf::RepeatedField" = type { i32, i32, %"struct.google::protobuf::RepeatedField<unsigned long>::Rep"* }
%"struct.google::protobuf::RepeatedField<unsigned long>::Rep" = type { %"class.google::protobuf::Arena"*, [1 x i64] }
%"class.std::vector.33" = type { %"struct.std::_Vector_base.34" }
%"struct.std::_Vector_base.34" = type { %"struct.std::_Vector_base<edata, std::allocator<edata> >::_Vector_impl" }
%"struct.std::_Vector_base<edata, std::allocator<edata> >::_Vector_impl" = type { %struct.edata*, %struct.edata*, %struct.edata* }
%struct.edata = type { i32, i8*, double, %"class.std::vector.38"*, %"class.std::vector.38"* }
%"class.std::map.71" = type { %"class.std::_Rb_tree.72" }
%"class.std::_Rb_tree.72" = type { %"struct.std::_Rb_tree<tfk::Tile *, std::pair<tfk::Tile *const, std::vector<float, std::allocator<float> > >, std::_Select1st<std::pair<tfk::Tile *const, std::vector<float, std::allocator<float> > > >, std::less<tfk::Tile *>, std::allocator<std::pair<tfk::Tile *const, std::vector<float, std::allocator<float> > > > >::_Rb_tree_impl" }
%"struct.std::_Rb_tree<tfk::Tile *, std::pair<tfk::Tile *const, std::vector<float, std::allocator<float> > >, std::_Select1st<std::pair<tfk::Tile *const, std::vector<float, std::allocator<float> > > >, std::less<tfk::Tile *>, std::allocator<std::pair<tfk::Tile *const, std::vector<float, std::allocator<float> > > > >::_Rb_tree_impl" = type { %"struct.std::less.76", %"struct.std::_Rb_tree_node_base", i64 }
%"class.std::map.78" = type { %"class.std::_Rb_tree.79" }
%"class.std::_Rb_tree.79" = type { %"struct.std::_Rb_tree<tfk::Tile *, std::pair<tfk::Tile *const, bool>, std::_Select1st<std::pair<tfk::Tile *const, bool> >, std::less<tfk::Tile *>, std::allocator<std::pair<tfk::Tile *const, bool> > >::_Rb_tree_impl" }
%"struct.std::_Rb_tree<tfk::Tile *, std::pair<tfk::Tile *const, bool>, std::_Select1st<std::pair<tfk::Tile *const, bool> >, std::less<tfk::Tile *>, std::allocator<std::pair<tfk::Tile *const, bool> > >::_Rb_tree_impl" = type { %"struct.std::less.76", %"struct.std::_Rb_tree_node_base", i64 }
%"class.std::map.83" = type { %"class.std::_Rb_tree.84" }
%"class.std::_Rb_tree.84" = type { %"struct.std::_Rb_tree<tfk::Tile *, std::pair<tfk::Tile *const, int>, std::_Select1st<std::pair<tfk::Tile *const, int> >, std::less<tfk::Tile *>, std::allocator<std::pair<tfk::Tile *const, int> > >::_Rb_tree_impl" }
%"struct.std::_Rb_tree<tfk::Tile *, std::pair<tfk::Tile *const, int>, std::_Select1st<std::pair<tfk::Tile *const, int> >, std::less<tfk::Tile *>, std::allocator<std::pair<tfk::Tile *const, int> > >::_Rb_tree_impl" = type { %"struct.std::less.76", %"struct.std::_Rb_tree_node_base", i64 }
%"class.std::mutex" = type { %"class.std::__mutex_base" }
%"class.std::__mutex_base" = type { %union.pthread_mutex_t }
%"class.std::vector.38" = type { %"struct.std::_Vector_base.39" }
%"struct.std::_Vector_base.39" = type { %"struct.std::_Vector_base<cv::Point_<float>, std::allocator<cv::Point_<float> > >::_Vector_impl" }
%"struct.std::_Vector_base<cv::Point_<float>, std::allocator<cv::Point_<float> > >::_Vector_impl" = type { %"class.cv::Point_"*, %"class.cv::Point_"*, %"class.cv::Point_"* }
%"class.std::set" = type { %"class.std::_Rb_tree.105" }
%"class.std::_Rb_tree.105" = type { %"struct.std::_Rb_tree<int, int, std::_Identity<int>, std::less<int>, std::allocator<int> >::_Rb_tree_impl" }
%"struct.std::_Rb_tree<int, int, std::_Identity<int>, std::less<int>, std::allocator<int> >::_Rb_tree_impl" = type { %"struct.std::less", %"struct.std::_Rb_tree_node_base", i64 }
%"class.std::vector.109" = type { %"struct.std::_Vector_base.110" }
%"struct.std::_Vector_base.110" = type { %"struct.std::_Vector_base<tfkMatch, std::allocator<tfkMatch> >::_Vector_impl" }
%"struct.std::_Vector_base<tfkMatch, std::allocator<tfkMatch> >::_Vector_impl" = type { %struct.tfkMatch*, %struct.tfkMatch*, %struct.tfkMatch* }
%struct.tfkMatch = type { %struct.tfkTriangle, %struct.tfkTriangle, [3 x double], [3 x double], %struct.graph_section_data, %struct.graph_section_data, i8*, i8*, %"class.cv::Point_" }
%struct.tfkTriangle = type { i32, i32, i32 }
%struct.graph_section_data = type { %"class.std::vector.38"*, %"class.std::vector.38"*, %"class.std::vector.38"*, %"class.std::vector.88"*, %"class.std::vector.94"*, %"class.cv::Mat"*, i32, %"class.cv::Point_"*, %"class.cv::Point_"*, double*, double* }
%"class.std::vector.88" = type { %"struct.std::_Vector_base.89" }
%"struct.std::_Vector_base.89" = type { %"struct.std::_Vector_base<std::pair<int, int>, std::allocator<std::pair<int, int> > >::_Vector_impl" }
%"struct.std::_Vector_base<std::pair<int, int>, std::allocator<std::pair<int, int> > >::_Vector_impl" = type { %"struct.std::pair.93"*, %"struct.std::pair.93"*, %"struct.std::pair.93"* }
%"struct.std::pair.93" = type { i32, i32 }
%"class.std::vector.94" = type opaque
%"class.tfk::TriangleMesh" = type { %"struct.std::pair", %"class.std::vector.38"*, %"class.std::vector.38"*, %"class.std::vector.88"*, %"class.std::vector.94"*, %"class.tfk::RangeTree"*, %"class.tfk::RangeTree"* }
%"class.tfk::RangeTree" = type { %"struct.tfk::Triangle"*, i32, i8, %"struct.std::pair", %"class.std::vector.114" }
%"struct.tfk::Triangle" = type { i32, [3 x %"class.cv::Point_"] }
%"class.std::vector.114" = type { %"struct.std::_Vector_base.115" }
%"struct.std::_Vector_base.115" = type { %"struct.std::_Vector_base<tfk::RangeTree *, std::allocator<tfk::RangeTree *> >::_Vector_impl" }
%"struct.std::_Vector_base<tfk::RangeTree *, std::allocator<tfk::RangeTree *> >::_Vector_impl" = type { %"class.tfk::RangeTree"**, %"class.tfk::RangeTree"**, %"class.tfk::RangeTree"** }
%class.Graph = type { %struct.vdata*, i32*, i64*, i32, i32, %"class.std::vector.100" }
%struct.vdata = type <{ i32, i32, i32, i32, i8*, double, double, double, double, %struct.graph_section_data*, %"class.std::vector.95"*, %"class.cv::Point_", %"class.cv::Point_", i32, i8, [3 x i8], double, double, double, double, double, double, i32, [4 x i8] }>
%"class.std::vector.100" = type { %"struct.std::_Vector_base.101" }
%"struct.std::_Vector_base.101" = type { %"struct.std::_Vector_base<std::vector<edata, std::allocator<edata> >, std::allocator<std::vector<edata, std::allocator<edata> > > >::_Vector_impl" }
%"struct.std::_Vector_base<std::vector<edata, std::allocator<edata> >, std::allocator<std::vector<edata, std::allocator<edata> > > >::_Vector_impl" = type { %"class.std::vector.33"*, %"class.std::vector.33"*, %"class.std::vector.33"* }
%struct._align_data = type { i32, i8*, i8*, i8*, i8*, i32, i32, i8, i32, i32, i32, i32, i8, float, i8, i8, i8, %"struct.std::pair" }
%struct.__va_list_tag = type { i32, i32, i8*, i8* }
%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep" = type { %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep_base" }
%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep_base" = type { i64, i64, i32 }
%"class.std::allocator" = type { i8 }

@.str.4 = private unnamed_addr constant [2 x i8] c"_\00", align 1
@.str.5 = private unnamed_addr constant [5 x i8] c".tif\00", align 1
@.str.16 = private unnamed_addr constant [3 x i8] c"%d\00", align 1
@_ZNSs4_Rep20_S_empty_rep_storageE = external global [0 x i64], align 8
@str = private unnamed_addr constant [80 x i8] c"Less than two images, so we're going to just render the stack without patching.\00"

declare void @_ZN2cv3Mat10deallocateEv(%"class.cv::Mat"*) local_unnamed_addr

declare void @_ZN2cv3Mat6createEiPKii(%"class.cv::Mat"*, i32, i32*, i32) local_unnamed_addr

declare void @_ZN2cv3Mat8copySizeERKS0_(%"class.cv::Mat"*, %"class.cv::Mat"* dereferenceable(96)) local_unnamed_addr

declare void @_ZN2cv6String10deallocateEv(%"class.cv::String"*) local_unnamed_addr

declare i8* @_ZN2cv6String8allocateEm(%"class.cv::String"*, i64) local_unnamed_addr

declare zeroext i1 @_ZN2cv7imwriteERKNS_6StringERKNS_11_InputArrayERKSt6vectorIiSaIiEE(%"class.cv::String"* dereferenceable(16), %"class.cv::_InputArray"* dereferenceable(24), %"class.std::vector.95"* dereferenceable(24)) local_unnamed_addr

declare void @_ZN2cv8fastFreeEPv(i8*) local_unnamed_addr

declare void @_ZN3tfk6Render12render_stackEPNS_5StackESt4pairIN2cv6Point_IfEES6_ENS_10ResolutionESs(%"class.tfk::Render"* readnone, %"class.tfk::Stack"* nocapture readonly, %"struct.std::pair"* nocapture readonly, i32, %"class.std::basic_string"*) local_unnamed_addr

declare void @_ZN3tfk6Render6renderEPNS_7SectionESt4pairIN2cv6Point_IfEES6_ENS_10ResolutionESs(%"class.tfk::Render"* readnone, %"class.tfk::Section"*, %"struct.std::pair"* nocapture readonly, i32, %"class.std::basic_string"* nocapture readonly) local_unnamed_addr

declare void @_ZN3tfk6Render6renderEPNS_7SectionESt4pairIN2cv6Point_IfEES6_ENS_10ResolutionEb(%"class.cv::Mat"* noalias sret, %"class.tfk::Render"* readnone, %"class.tfk::Section"*, %"struct.std::pair"* nocapture readonly, i32, i1 zeroext) local_unnamed_addr

declare void @_ZN9__gnu_cxx12__to_xstringISscEET_PFiPT0_mPKS2_P13__va_list_tagEmS5_z(%"class.std::basic_string"* noalias sret, i32 (i8*, i64, i8*, %struct.__va_list_tag*)*, i64, i8*, ...) local_unnamed_addr

declare void @_ZNSs4_Rep10_M_destroyERKSaIcE(%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"*, %"class.std::allocator"* dereferenceable(1)) local_unnamed_addr

declare dereferenceable(8) %"class.std::basic_string"* @_ZNSs6appendEPKcm(%"class.std::basic_string"*, i8*, i64) local_unnamed_addr

declare dereferenceable(8) %"class.std::basic_string"* @_ZNSs6appendERKSs(%"class.std::basic_string"*, %"class.std::basic_string"* dereferenceable(8)) local_unnamed_addr

declare dereferenceable(8) %"class.std::basic_string"* @_ZNSs6insertEmPKcm(%"class.std::basic_string"*, i64, i8*, i64) local_unnamed_addr

declare void @_ZNSsC1ERKSs(%"class.std::basic_string"*, %"class.std::basic_string"* dereferenceable(8)) unnamed_addr

declare i32 @__gxx_personality_v0(...)

declare void @_ZdlPv(i8*) local_unnamed_addr

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #0

declare hidden void @__clang_call_terminate(i8*) local_unnamed_addr

declare extern_weak i32 @__pthread_key_create(i32*, void (i8*)*)

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #0

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #0

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1) #0

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i32, i1) #0

declare i32 @puts(i8* nocapture readonly) local_unnamed_addr

declare i32 @vsnprintf(i8* nocapture, i64, i8* nocapture readonly, %struct.__va_list_tag*)

define void @_ZN3tfk6Render23render_stack_with_patchEPNS_5StackESt4pairIN2cv6Point_IfEES6_ENS_10ResolutionESs(%"class.tfk::Render"* readnone %this, %"class.tfk::Stack"* nocapture readonly %stack, %"struct.std::pair"* nocapture readonly %bbox, i32 %resolution, %"class.std::basic_string"* %filename_prefix) local_unnamed_addr align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %ref.tmp.i1317 = alloca %"class.std::allocator", align 1
  %ref.tmp.i1303 = alloca %"class.std::allocator", align 1
  %ref.tmp.i1289 = alloca %"class.std::allocator", align 1
  %ref.tmp.i1275 = alloca %"class.std::allocator", align 1
  %ref.tmp.i1101 = alloca %"class.std::allocator", align 1
  %ref.tmp.i1087 = alloca %"class.std::allocator", align 1
  %ref.tmp.i1073 = alloca %"class.std::allocator", align 1
  %ref.tmp.i1059 = alloca %"class.std::allocator", align 1
  %ref.tmp.i.i1002 = alloca %"class.std::allocator", align 1
  %ref.tmp.i.i = alloca %"class.std::allocator", align 1
  %ref.tmp.i805 = alloca %"class.std::allocator", align 1
  %ref.tmp.i784 = alloca %"class.std::allocator", align 1
  %ref.tmp.i763 = alloca %"class.std::allocator", align 1
  %ref.tmp.i742 = alloca %"class.std::allocator", align 1
  %ref.tmp.i721 = alloca %"class.std::allocator", align 1
  %ref.tmp.i700 = alloca %"class.std::allocator", align 1
  %sz.i = alloca [2 x i32], align 4
  %ref.tmp.i659 = alloca %"class.std::allocator", align 1
  %ref.tmp.i639 = alloca %"class.std::allocator", align 1
  %ref.tmp.i552 = alloca %"class.std::allocator", align 1
  %ref.tmp.i = alloca %"class.std::allocator", align 1
  %agg.tmp = alloca <4 x i32>, align 16
  %agg.tmp3 = alloca %"class.std::basic_string", align 8
  %next_section_img = alloca %"class.cv::Mat", align 16
  %agg.tmp6 = alloca <4 x i32>, align 16
  %img = alloca %"class.cv::Mat", align 8
  %agg.tmp9 = alloca <4 x i32>, align 16
  %last_img = alloca %"class.cv::Mat", align 8
  %section_p_out_sum = alloca %"class.cv::Mat", align 8
  %section_p_out_count = alloca %"class.cv::Mat", align 8
  %syncreg203 = tail call token @llvm.syncregion.start()
  %ref.tmp = alloca %"class.cv::String", align 8
  %ref.tmp301 = alloca %"class.std::basic_string", align 8
  %ref.tmp302 = alloca %"class.std::basic_string", align 8
  %ref.tmp305 = alloca %"class.std::basic_string", align 8
  %ref.tmp314 = alloca %"class.cv::_InputArray", align 8
  %ref.tmp317 = alloca %"class.std::vector.95", align 8
  %ref.tmp345 = alloca %"class.cv::Mat", align 16
  %agg.tmp350 = alloca <4 x i32>, align 16
  %tmpcast1941 = bitcast <4 x i32>* %agg.tmp350 to %"struct.std::pair"*
  %agg.tmp381 = alloca <4 x i32>, align 16
  %tmpcast1940 = bitcast <4 x i32>* %agg.tmp381 to %"struct.std::pair"*
  %agg.tmp384 = alloca %"class.std::basic_string", align 8
  %ref.tmp385 = alloca %"class.std::basic_string", align 8
  %ref.tmp386 = alloca %"class.std::basic_string", align 8
  %ref.tmp389 = alloca %"class.std::basic_string", align 8
  %sections = getelementptr inbounds %"class.tfk::Stack", %"class.tfk::Stack"* %stack, i64 0, i32 12
  %_M_finish.i = getelementptr inbounds %"class.tfk::Stack", %"class.tfk::Stack"* %stack, i64 0, i32 12, i32 0, i32 0, i32 1
  %0 = bitcast %"class.tfk::Section"*** %_M_finish.i to i64*
  %1 = load i64, i64* %0, align 8, !tbaa !0
  %2 = bitcast %"class.std::vector.129"* %sections to i64*
  %3 = load i64, i64* %2, align 8, !tbaa !6
  %sub.ptr.sub.i = sub i64 %1, %3
  %sub.ptr.div.i = ashr exact i64 %sub.ptr.sub.i, 3
  %cmp = icmp ult i64 %sub.ptr.div.i, 2
  br i1 %cmp, label %if.then, label %invoke.cont11

if.then:                                          ; preds = %entry
  %tmpcast1942 = bitcast <4 x i32>* %agg.tmp to %"struct.std::pair"*
  %puts = tail call i32 @puts(i8* getelementptr inbounds ([80 x i8], [80 x i8]* @str, i64 0, i64 0))
  %4 = bitcast %"struct.std::pair"* %bbox to <4 x i32>*
  %5 = load <4 x i32>, <4 x i32>* %4, align 4, !tbaa !7
  store <4 x i32> %5, <4 x i32>* %agg.tmp, align 16, !tbaa !7
  call void @_ZNSsC1ERKSs(%"class.std::basic_string"* nonnull %agg.tmp3, %"class.std::basic_string"* dereferenceable(8) %filename_prefix)
  invoke void @_ZN3tfk6Render12render_stackEPNS_5StackESt4pairIN2cv6Point_IfEES6_ENS_10ResolutionESs(%"class.tfk::Render"* %this, %"class.tfk::Stack"* nonnull %stack, %"struct.std::pair"* nonnull %tmpcast1942, i32 %resolution, %"class.std::basic_string"* nonnull %agg.tmp3)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %if.then
  %_M_p.i.i.i = getelementptr inbounds %"class.std::basic_string", %"class.std::basic_string"* %agg.tmp3, i64 0, i32 0, i32 0
  %6 = load i8*, i8** %_M_p.i.i.i, align 8, !tbaa !9
  %arrayidx.i.i549 = getelementptr inbounds i8, i8* %6, i64 -24
  %7 = bitcast i8* %arrayidx.i.i549 to %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"*
  %8 = getelementptr inbounds %"class.std::allocator", %"class.std::allocator"* %ref.tmp.i, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %8)
  %cmp.i.i550 = icmp eq i8* %arrayidx.i.i549, bitcast ([0 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE to i8*)
  br i1 %cmp.i.i550, label %_ZNSsD2Ev.exit, label %if.then.i.i551, !prof !12

if.then.i.i551:                                   ; preds = %invoke.cont
  %_M_refcount.i.i = getelementptr inbounds i8, i8* %6, i64 -8
  %9 = bitcast i8* %_M_refcount.i.i to i32*
  br i1 icmp ne (i8* bitcast (i32 (i32*, void (i8*)*)* @__pthread_key_create to i8*), i8* null), label %if.then.i.i.i, label %if.else.i.i.i

if.then.i.i.i:                                    ; preds = %if.then.i.i551
  %10 = atomicrmw volatile add i32* %9, i32 -1 acq_rel
  br label %invoke.cont.i.i

if.else.i.i.i:                                    ; preds = %if.then.i.i551
  %11 = load i32, i32* %9, align 4, !tbaa !13
  %add.i.i.i.i = add nsw i32 %11, -1
  store i32 %add.i.i.i.i, i32* %9, align 4, !tbaa !13
  br label %invoke.cont.i.i

invoke.cont.i.i:                                  ; preds = %if.else.i.i.i, %if.then.i.i.i
  %retval.0.i.i.i = phi i32 [ %10, %if.then.i.i.i ], [ %11, %if.else.i.i.i ]
  %cmp3.i.i = icmp slt i32 %retval.0.i.i.i, 1
  br i1 %cmp3.i.i, label %if.then4.i.i, label %_ZNSsD2Ev.exit

if.then4.i.i:                                     ; preds = %invoke.cont.i.i
  call void @_ZNSs4_Rep10_M_destroyERKSaIcE(%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"* nonnull %7, %"class.std::allocator"* nonnull dereferenceable(1) %ref.tmp.i)
  br label %_ZNSsD2Ev.exit

_ZNSsD2Ev.exit:                                   ; preds = %if.then4.i.i, %invoke.cont.i.i, %invoke.cont
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %8)
  br label %cleanup.cont

lpad:                                             ; preds = %if.then
  %12 = landingpad { i8*, i32 }
          cleanup
  %13 = extractvalue { i8*, i32 } %12, 0
  %14 = extractvalue { i8*, i32 } %12, 1
  %_M_p.i.i.i553 = getelementptr inbounds %"class.std::basic_string", %"class.std::basic_string"* %agg.tmp3, i64 0, i32 0, i32 0
  %15 = load i8*, i8** %_M_p.i.i.i553, align 8, !tbaa !9
  %arrayidx.i.i554 = getelementptr inbounds i8, i8* %15, i64 -24
  %16 = bitcast i8* %arrayidx.i.i554 to %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"*
  %17 = getelementptr inbounds %"class.std::allocator", %"class.std::allocator"* %ref.tmp.i552, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %17)
  %cmp.i.i555 = icmp eq i8* %arrayidx.i.i554, bitcast ([0 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE to i8*)
  br i1 %cmp.i.i555, label %_ZNSsD2Ev.exit565, label %if.then.i.i557, !prof !12

if.then.i.i557:                                   ; preds = %lpad
  %_M_refcount.i.i556 = getelementptr inbounds i8, i8* %15, i64 -8
  %18 = bitcast i8* %_M_refcount.i.i556 to i32*
  br i1 icmp ne (i8* bitcast (i32 (i32*, void (i8*)*)* @__pthread_key_create to i8*), i8* null), label %if.then.i.i.i558, label %if.else.i.i.i560

if.then.i.i.i558:                                 ; preds = %if.then.i.i557
  %19 = atomicrmw volatile add i32* %18, i32 -1 acq_rel
  br label %invoke.cont.i.i563

if.else.i.i.i560:                                 ; preds = %if.then.i.i557
  %20 = load i32, i32* %18, align 4, !tbaa !13
  %add.i.i.i.i559 = add nsw i32 %20, -1
  store i32 %add.i.i.i.i559, i32* %18, align 4, !tbaa !13
  br label %invoke.cont.i.i563

invoke.cont.i.i563:                               ; preds = %if.else.i.i.i560, %if.then.i.i.i558
  %retval.0.i.i.i561 = phi i32 [ %19, %if.then.i.i.i558 ], [ %20, %if.else.i.i.i560 ]
  %cmp3.i.i562 = icmp slt i32 %retval.0.i.i.i561, 1
  br i1 %cmp3.i.i562, label %if.then4.i.i564, label %_ZNSsD2Ev.exit565

if.then4.i.i564:                                  ; preds = %invoke.cont.i.i563
  call void @_ZNSs4_Rep10_M_destroyERKSaIcE(%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"* nonnull %16, %"class.std::allocator"* nonnull dereferenceable(1) %ref.tmp.i552)
  br label %_ZNSsD2Ev.exit565

_ZNSsD2Ev.exit565:                                ; preds = %if.then4.i.i564, %invoke.cont.i.i563, %lpad
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %17)
  br label %eh.resume

invoke.cont11:                                    ; preds = %entry
  %tmpcast1939 = bitcast <4 x i32>* %agg.tmp9 to %"struct.std::pair"*
  %tmpcast = bitcast <4 x i32>* %agg.tmp6 to %"struct.std::pair"*
  %21 = inttoptr i64 %3 to %"class.tfk::Section"**
  %_M_start.i = getelementptr inbounds %"class.std::vector.129", %"class.std::vector.129"* %sections, i64 0, i32 0, i32 0, i32 0
  %add.ptr.i = getelementptr inbounds %"class.tfk::Section"*, %"class.tfk::Section"** %21, i64 1
  %22 = load %"class.tfk::Section"*, %"class.tfk::Section"** %add.ptr.i, align 8, !tbaa !15
  %23 = bitcast %"class.cv::Mat"* %next_section_img to i8*
  call void @llvm.lifetime.start.p0i8(i64 96, i8* nonnull %23)
  %24 = bitcast %"struct.std::pair"* %bbox to <4 x i32>*
  %25 = load <4 x i32>, <4 x i32>* %24, align 4, !tbaa !7
  store <4 x i32> %25, <4 x i32>* %agg.tmp6, align 16, !tbaa !7
  call void @_ZN3tfk6Render6renderEPNS_7SectionESt4pairIN2cv6Point_IfEES6_ENS_10ResolutionEb(%"class.cv::Mat"* nonnull sret %next_section_img, %"class.tfk::Render"* %this, %"class.tfk::Section"* %22, %"struct.std::pair"* nonnull %tmpcast, i32 %resolution, i1 zeroext false)
  %26 = bitcast %"class.cv::Mat"* %img to i8*
  call void @llvm.lifetime.start.p0i8(i64 96, i8* nonnull %26)
  %27 = load %"class.tfk::Section"**, %"class.tfk::Section"*** %_M_start.i, align 8, !tbaa !6
  %28 = load %"class.tfk::Section"*, %"class.tfk::Section"** %27, align 8, !tbaa !15
  %29 = bitcast %"struct.std::pair"* %bbox to <4 x i32>*
  %30 = load <4 x i32>, <4 x i32>* %29, align 4, !tbaa !7
  store <4 x i32> %30, <4 x i32>* %agg.tmp9, align 16, !tbaa !7
  invoke void @_ZN3tfk6Render6renderEPNS_7SectionESt4pairIN2cv6Point_IfEES6_ENS_10ResolutionEb(%"class.cv::Mat"* nonnull sret %img, %"class.tfk::Render"* %this, %"class.tfk::Section"* %28, %"struct.std::pair"* nonnull %tmpcast1939, i32 %resolution, i1 zeroext false)
          to label %invoke.cont14 unwind label %lpad10

invoke.cont14:                                    ; preds = %invoke.cont11
  %31 = bitcast %"class.cv::Mat"* %last_img to i8*
  call void @llvm.lifetime.start.p0i8(i64 96, i8* nonnull %31)
  %flags.i = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %last_img, i64 0, i32 0
  store i32 1124007936, i32* %flags.i, align 8, !tbaa !16
  %dims.i = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %last_img, i64 0, i32 1
  %rows.i = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %last_img, i64 0, i32 2
  %p.i.i603 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %last_img, i64 0, i32 10, i32 0
  %32 = bitcast i32* %dims.i to i8*
  call void @llvm.memset.p0i8.i64(i8* nonnull %32, i8 0, i64 60, i32 4, i1 false)
  store i32* %rows.i, i32** %p.i.i603, align 8, !tbaa !20
  %arraydecay.i.i = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %last_img, i64 0, i32 11, i32 1, i64 0
  %p.i3.i = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %last_img, i64 0, i32 11, i32 0
  store i64* %arraydecay.i.i, i64** %p.i3.i, align 8, !tbaa !21
  %33 = bitcast i64* %arraydecay.i.i to i8*
  call void @llvm.memset.p0i8.i64(i8* nonnull %33, i8 0, i64 16, i32 8, i1 false)
  %34 = load i64, i64* %0, align 8, !tbaa !0
  %35 = load i64, i64* %2, align 8, !tbaa !6
  %cmp171824 = icmp eq i64 %34, %35
  br i1 %cmp171824, label %for.cond.cleanup375, label %invoke.cont24.lr.ph

invoke.cont24.lr.ph:                              ; preds = %invoke.cont14
  %36 = bitcast %"class.cv::Mat"* %section_p_out_sum to i8*
  %flags.i633 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %section_p_out_sum, i64 0, i32 0
  %dims.i634 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %section_p_out_sum, i64 0, i32 1
  %rows.i635 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %section_p_out_sum, i64 0, i32 2
  %p.i.i636 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %section_p_out_sum, i64 0, i32 10, i32 0
  %37 = bitcast i32* %dims.i634 to i8*
  %arraydecay.i.i637 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %section_p_out_sum, i64 0, i32 11, i32 1, i64 0
  %p.i3.i638 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %section_p_out_sum, i64 0, i32 11, i32 0
  %38 = bitcast i64* %arraydecay.i.i637 to i8*
  %39 = bitcast %"class.cv::Mat"* %section_p_out_count to i8*
  %flags.i653 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %section_p_out_count, i64 0, i32 0
  %dims.i654 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %section_p_out_count, i64 0, i32 1
  %rows.i655 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %section_p_out_count, i64 0, i32 2
  %p.i.i656 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %section_p_out_count, i64 0, i32 10, i32 0
  %40 = bitcast i32* %dims.i654 to i8*
  %arraydecay.i.i657 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %section_p_out_count, i64 0, i32 11, i32 1, i64 0
  %p.i3.i658 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %section_p_out_count, i64 0, i32 11, i32 0
  %41 = bitcast i64* %arraydecay.i.i657 to i8*
  %rows = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %img, i64 0, i32 2
  %cols = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %img, i64 0, i32 3
  %42 = bitcast [2 x i32]* %sz.i to i8*
  %arrayinit.begin.i = getelementptr inbounds [2 x i32], [2 x i32]* %sz.i, i64 0, i64 0
  %arrayinit.element.i = getelementptr inbounds [2 x i32], [2 x i32]* %sz.i, i64 0, i64 1
  %cols.i682 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %section_p_out_count, i64 0, i32 3
  %data.i697 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %section_p_out_sum, i64 0, i32 4
  %data.i714 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %section_p_out_count, i64 0, i32 4
  %data.i735 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %img, i64 0, i32 4
  %p.i736 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %img, i64 0, i32 11, i32 0
  %data.i756 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %next_section_img, i64 0, i32 4
  %p.i757 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %next_section_img, i64 0, i32 11, i32 0
  %data.i826 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %last_img, i64 0, i32 4
  %43 = bitcast %"class.cv::String"* %ref.tmp to i8*
  %44 = bitcast %"class.std::basic_string"* %ref.tmp301 to i8*
  %45 = bitcast %"class.std::basic_string"* %ref.tmp302 to i8*
  %46 = bitcast %"class.std::basic_string"* %ref.tmp305 to i8*
  %_M_p.i.i.i.i1024 = getelementptr inbounds %"class.std::basic_string", %"class.std::basic_string"* %ref.tmp302, i64 0, i32 0, i32 0
  %_M_p.i.i.i17.i1027 = getelementptr inbounds %"class.std::basic_string", %"class.std::basic_string"* %ref.tmp305, i64 0, i32 0, i32 0
  %47 = bitcast %"class.std::basic_string"* %ref.tmp301 to i64*
  %48 = bitcast %"class.cv::_InputArray"* %ref.tmp314 to i8*
  %width.i.i = getelementptr inbounds %"class.cv::_InputArray", %"class.cv::_InputArray"* %ref.tmp314, i64 0, i32 2, i32 0
  %height.i.i = getelementptr inbounds %"class.cv::_InputArray", %"class.cv::_InputArray"* %ref.tmp314, i64 0, i32 2, i32 1
  %flags.i.i1053 = getelementptr inbounds %"class.cv::_InputArray", %"class.cv::_InputArray"* %ref.tmp314, i64 0, i32 0
  %obj.i.i = getelementptr inbounds %"class.cv::_InputArray", %"class.cv::_InputArray"* %ref.tmp314, i64 0, i32 1
  %49 = bitcast i8** %obj.i.i to %"class.cv::Mat"**
  %50 = bitcast %"class.std::vector.95"* %ref.tmp317 to i8*
  %_M_start.i.i = getelementptr inbounds %"class.std::vector.95", %"class.std::vector.95"* %ref.tmp317, i64 0, i32 0, i32 0, i32 0
  %51 = getelementptr inbounds %"class.std::allocator", %"class.std::allocator"* %ref.tmp.i1059, i64 0, i32 0
  %_M_p.i.i.i1074 = getelementptr inbounds %"class.std::basic_string", %"class.std::basic_string"* %ref.tmp301, i64 0, i32 0, i32 0
  %52 = getelementptr inbounds %"class.std::allocator", %"class.std::allocator"* %ref.tmp.i1073, i64 0, i32 0
  %53 = getelementptr inbounds %"class.std::allocator", %"class.std::allocator"* %ref.tmp.i1087, i64 0, i32 0
  %54 = getelementptr inbounds %"class.std::allocator", %"class.std::allocator"* %ref.tmp.i1101, i64 0, i32 0
  %u.i = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %img, i64 0, i32 9
  %u.i.i1117 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %last_img, i64 0, i32 9
  %55 = bitcast i8** %data.i826 to i8*
  %dims6.i = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %img, i64 0, i32 1
  %allocator.i = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %img, i64 0, i32 8
  %56 = bitcast %"class.cv::MatAllocator"** %allocator.i to i64*
  %allocator24.i = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %last_img, i64 0, i32 8
  %57 = bitcast %"class.cv::MatAllocator"** %allocator24.i to i64*
  %58 = bitcast %"struct.cv::UMatData"** %u.i to i64*
  %59 = bitcast %"struct.cv::UMatData"** %u.i.i1117 to i64*
  %u.i1142 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %next_section_img, i64 0, i32 9
  %flags50.i = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %img, i64 0, i32 0
  %cols12.i = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %last_img, i64 0, i32 3
  %60 = bitcast i8** %data.i735 to i8*
  %p.i.i1161 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %img, i64 0, i32 10, i32 0
  %dims6.i1172 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %next_section_img, i64 0, i32 1
  %allocator.i1192 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %next_section_img, i64 0, i32 8
  %61 = bitcast %"class.cv::MatAllocator"** %allocator.i1192 to i64*
  %62 = bitcast %"struct.cv::UMatData"** %u.i1142 to i64*
  %flags50.i1158 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %next_section_img, i64 0, i32 0
  %rows.i1175 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %next_section_img, i64 0, i32 2
  %cols.i1177 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %next_section_img, i64 0, i32 3
  %63 = bitcast %"class.cv::Mat"* %ref.tmp345 to i8*
  %u.i.i1356 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %section_p_out_count, i64 0, i32 9
  %64 = bitcast i8** %data.i714 to i8*
  %65 = bitcast i8** %data.i756 to i8*
  %p.i.i1219 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %next_section_img, i64 0, i32 10, i32 0
  %flags.i1226 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %ref.tmp345, i64 0, i32 0
  %dims.i1227 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %ref.tmp345, i64 0, i32 1
  %rows.i1228 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %ref.tmp345, i64 0, i32 2
  %data.i1230 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %ref.tmp345, i64 0, i32 4
  %allocator.i1234 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %ref.tmp345, i64 0, i32 8
  %66 = bitcast %"class.cv::MatAllocator"** %allocator.i1234 to i64*
  %u.i1235 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %ref.tmp345, i64 0, i32 9
  %67 = bitcast %"struct.cv::UMatData"** %u.i1235 to i64*
  %p.i1236 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %next_section_img, i64 0, i32 11, i32 0
  %arraydecay.i1237 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %next_section_img, i64 0, i32 11, i32 1, i64 0
  %u.i.i1381 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %section_p_out_sum, i64 0, i32 9
  %p.i90.i = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %ref.tmp345, i64 0, i32 11, i32 0
  %68 = bitcast i8** %data.i697 to i8*
  %69 = bitcast i32* %dims.i1227 to i8*
  %p40.i = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %ref.tmp345, i64 0, i32 10, i32 0
  %size41.i = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %next_section_img, i64 0, i32 10
  %arraydecay45.i = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %ref.tmp345, i64 0, i32 11, i32 1, i64 0
  %70 = bitcast i8** %data.i1230 to i8*
  %p.i1259 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %ref.tmp345, i64 0, i32 11, i32 0
  %.pre = load %"class.tfk::Section"**, %"class.tfk::Section"*** %_M_start.i, align 8, !tbaa !6
  %arrayidx.i92.i = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %next_section_img, i64 0, i32 11, i32 1, i64 1
  %71 = bitcast i8** %data.i735 to <4 x i64>*
  %72 = bitcast i8** %data.i826 to <4 x i64>*
  %73 = bitcast i8** %data.i756 to <4 x i64>*
  %74 = bitcast i8** %data.i735 to <4 x i64>*
  %75 = bitcast %"struct.std::pair"* %bbox to <4 x i32>*
  %76 = bitcast %"class.cv::Mat"* %ref.tmp345 to <4 x i32>*
  %77 = bitcast %"class.cv::Mat"* %next_section_img to <4 x i32>*
  %78 = bitcast i8** %data.i1230 to <4 x i64>*
  %79 = bitcast i8** %data.i756 to <4 x i64>*
  %80 = bitcast i32** %p40.i to <2 x i64>*
  %81 = bitcast %"struct.cv::MatSize"* %size41.i to <2 x i64>*
  br label %invoke.cont24

for.cond.cleanup:                                 ; preds = %_ZN2cv3MatD2Ev.exit1405
  %cmp3741804 = icmp eq i64 %427, %428
  br i1 %cmp3741804, label %for.cond.cleanup375, label %invoke.cont383.lr.ph

invoke.cont383.lr.ph:                             ; preds = %for.cond.cleanup
  %82 = bitcast %"class.std::basic_string"* %ref.tmp385 to i8*
  %83 = bitcast %"class.std::basic_string"* %ref.tmp386 to i8*
  %84 = bitcast %"class.std::basic_string"* %ref.tmp389 to i8*
  %_M_p.i.i.i.i = getelementptr inbounds %"class.std::basic_string", %"class.std::basic_string"* %ref.tmp386, i64 0, i32 0, i32 0
  %_M_p.i.i.i17.i = getelementptr inbounds %"class.std::basic_string", %"class.std::basic_string"* %ref.tmp389, i64 0, i32 0, i32 0
  %85 = bitcast %"class.std::basic_string"* %ref.tmp385 to i64*
  %86 = bitcast %"class.std::basic_string"* %agg.tmp384 to i64*
  %87 = getelementptr inbounds %"class.std::allocator", %"class.std::allocator"* %ref.tmp.i805, i64 0, i32 0
  %_M_p.i.i.i785 = getelementptr inbounds %"class.std::basic_string", %"class.std::basic_string"* %ref.tmp385, i64 0, i32 0, i32 0
  %88 = getelementptr inbounds %"class.std::allocator", %"class.std::allocator"* %ref.tmp.i784, i64 0, i32 0
  %89 = getelementptr inbounds %"class.std::allocator", %"class.std::allocator"* %ref.tmp.i763, i64 0, i32 0
  %90 = getelementptr inbounds %"class.std::allocator", %"class.std::allocator"* %ref.tmp.i742, i64 0, i32 0
  %91 = bitcast %"struct.std::pair"* %bbox to <4 x i32>*
  br label %invoke.cont383

lpad10:                                           ; preds = %invoke.cont11
  %92 = landingpad { i8*, i32 }
          cleanup
  %93 = extractvalue { i8*, i32 } %92, 0
  %94 = extractvalue { i8*, i32 } %92, 1
  br label %ehcleanup414

invoke.cont24:                                    ; preds = %_ZN2cv3MatD2Ev.exit1405, %invoke.cont24.lr.ph
  %95 = phi %"class.tfk::Section"** [ %.pre, %invoke.cont24.lr.ph ], [ %429, %_ZN2cv3MatD2Ev.exit1405 ]
  %indvars.iv1884 = phi i64 [ 0, %invoke.cont24.lr.ph ], [ %indvars.iv.next1885, %_ZN2cv3MatD2Ev.exit1405 ]
  %add.ptr.i632 = getelementptr inbounds %"class.tfk::Section"*, %"class.tfk::Section"** %95, i64 %indvars.iv1884
  %96 = load %"class.tfk::Section"*, %"class.tfk::Section"** %add.ptr.i632, align 8, !tbaa !15
  call void @llvm.lifetime.start.p0i8(i64 96, i8* nonnull %36)
  store i32 1124007936, i32* %flags.i633, align 8, !tbaa !16
  call void @llvm.memset.p0i8.i64(i8* nonnull %37, i8 0, i64 60, i32 4, i1 false)
  store i32* %rows.i635, i32** %p.i.i636, align 8, !tbaa !20
  store i64* %arraydecay.i.i637, i64** %p.i3.i638, align 8, !tbaa !21
  call void @llvm.memset.p0i8.i64(i8* nonnull %38, i8 0, i64 16, i32 8, i1 false)
  call void @llvm.lifetime.start.p0i8(i64 96, i8* nonnull %39)
  store i32 1124007936, i32* %flags.i653, align 8, !tbaa !16
  call void @llvm.memset.p0i8.i64(i8* nonnull %40, i8 0, i64 60, i32 4, i1 false)
  store i32* %rows.i655, i32** %p.i.i656, align 8, !tbaa !20
  store i64* %arraydecay.i.i657, i64** %p.i3.i658, align 8, !tbaa !21
  call void @llvm.memset.p0i8.i64(i8* nonnull %41, i8 0, i64 16, i32 8, i1 false)
  %97 = load i32, i32* %rows, align 8, !tbaa !22
  %98 = load i32, i32* %cols, align 4, !tbaa !23
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %42)
  store i32 %97, i32* %arrayinit.begin.i, align 4, !tbaa !13
  store i32 %98, i32* %arrayinit.element.i, align 4, !tbaa !13
  invoke void @_ZN2cv3Mat6createEiPKii(%"class.cv::Mat"* nonnull %section_p_out_sum, i32 2, i32* nonnull %arrayinit.begin.i, i32 2)
          to label %invoke.cont26 unwind label %lpad25

invoke.cont26:                                    ; preds = %invoke.cont24
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %42)
  %99 = load i32, i32* %rows, align 8, !tbaa !22
  %100 = load i32, i32* %cols, align 4, !tbaa !23
  %101 = load i32, i32* %dims.i654, align 4, !tbaa !24
  %cmp.i678 = icmp slt i32 %101, 3
  %102 = load i32, i32* %rows.i655, align 8
  %cmp2.i680 = icmp eq i32 %102, %99
  %or.cond1489 = and i1 %cmp.i678, %cmp2.i680
  %103 = load i32, i32* %cols.i682, align 4
  %cmp4.i683 = icmp eq i32 %103, %100
  %or.cond = and i1 %or.cond1489, %cmp4.i683
  br i1 %or.cond, label %land.lhs.true5.i688, label %if.end.i694

land.lhs.true5.i688:                              ; preds = %invoke.cont26
  %104 = load i32, i32* %flags.i653, align 8, !tbaa !16
  %and.i.i686 = and i32 %104, 4095
  %cmp6.i687 = icmp ne i32 %and.i.i686, 2
  %105 = load i8*, i8** %data.i714, align 8
  %tobool.i690 = icmp eq i8* %105, null
  %or.cond1826 = or i1 %cmp6.i687, %tobool.i690
  br i1 %or.cond1826, label %if.end.i694, label %invoke.cont29

if.end.i694:                                      ; preds = %land.lhs.true5.i688, %invoke.cont26
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %42)
  store i32 %99, i32* %arrayinit.begin.i, align 4, !tbaa !13
  store i32 %100, i32* %arrayinit.element.i, align 4, !tbaa !13
  invoke void @_ZN2cv3Mat6createEiPKii(%"class.cv::Mat"* nonnull %section_p_out_count, i32 2, i32* nonnull %arrayinit.begin.i, i32 2)
          to label %.noexc695 unwind label %lpad25

.noexc695:                                        ; preds = %if.end.i694
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %42)
  %.pre1886 = load i32, i32* %cols, align 4, !tbaa !23
  br label %invoke.cont29

invoke.cont29:                                    ; preds = %.noexc695, %land.lhs.true5.i688
  %106 = phi i32 [ %100, %land.lhs.true5.i688 ], [ %.pre1886, %.noexc695 ]
  %cmp321808 = icmp sgt i32 %106, 0
  br i1 %cmp321808, label %pfor.detach.lr.ph, label %pfor.cond.cleanup

pfor.detach.lr.ph:                                ; preds = %invoke.cont29
  %107 = load i32, i32* %rows, align 8
  %cmp361806 = icmp sgt i32 %107, 0
  %108 = load i8*, i8** %data.i697, align 8
  %109 = load i8*, i8** %data.i714, align 8
  br i1 %cmp361806, label %pfor.detach.lr.ph.split.us, label %pfor.cond.cleanup

pfor.detach.lr.ph.split.us:                       ; preds = %pfor.detach.lr.ph
  %110 = load i64*, i64** %p.i3.i658, align 8
  %111 = load i64*, i64** %p.i3.i638, align 8
  %112 = load i64, i64* %111, align 8, !tbaa !25
  %113 = load i64, i64* %110, align 8, !tbaa !25
  %wide.trip.count = zext i32 %107 to i64
  %wide.trip.count1853 = zext i32 %106 to i64
  %114 = add nsw i64 %wide.trip.count, -1
  %xtraiter = and i64 %wide.trip.count, 3
  %115 = icmp ult i64 %114, 3
  %unroll_iter = sub nsw i64 %wide.trip.count, %xtraiter
  %lcmp.mod = icmp eq i64 %xtraiter, 0
  br label %pfor.detach.us

pfor.detach.us:                                   ; preds = %pfor.inc.us, %pfor.detach.lr.ph.split.us
  %indvars.iv1851 = phi i64 [ %indvars.iv.next1852, %pfor.inc.us ], [ 0, %pfor.detach.lr.ph.split.us ]
  detach within %syncreg203, label %for.body39.lr.ph.us, label %pfor.inc.us

pfor.inc.us:                                      ; preds = %for.cond34.for.cond.cleanup37_crit_edge.us, %pfor.detach.us
  %indvars.iv.next1852 = add nuw nsw i64 %indvars.iv1851, 1
  %exitcond1854 = icmp eq i64 %indvars.iv.next1852, %wide.trip.count1853
  br i1 %exitcond1854, label %pfor.cond.cleanup, label %pfor.detach.us, !llvm.loop !27

for.body39.us:                                    ; preds = %for.body39.lr.ph.us.new, %for.body39.us
  %indvars.iv1848 = phi i64 [ 0, %for.body39.lr.ph.us.new ], [ %indvars.iv.next1849.3, %for.body39.us ]
  %niter = phi i64 [ %unroll_iter, %for.body39.lr.ph.us.new ], [ %niter.nsub.3, %for.body39.us ]
  %mul.i.us = mul i64 %112, %indvars.iv1848
  %add.ptr.i699.us = getelementptr inbounds i8, i8* %108, i64 %mul.i.us
  %116 = bitcast i8* %add.ptr.i699.us to i16*
  %arrayidx2.i.us = getelementptr inbounds i16, i16* %116, i64 %indvars.iv1851
  store i16 0, i16* %arrayidx2.i.us, align 2, !tbaa !29
  %mul.i717.us = mul i64 %113, %indvars.iv1848
  %add.ptr.i718.us = getelementptr inbounds i8, i8* %109, i64 %mul.i717.us
  %117 = bitcast i8* %add.ptr.i718.us to i16*
  %arrayidx2.i720.us = getelementptr inbounds i16, i16* %117, i64 %indvars.iv1851
  store i16 0, i16* %arrayidx2.i720.us, align 2, !tbaa !29
  %indvars.iv.next1849 = or i64 %indvars.iv1848, 1
  %mul.i.us.1 = mul i64 %112, %indvars.iv.next1849
  %add.ptr.i699.us.1 = getelementptr inbounds i8, i8* %108, i64 %mul.i.us.1
  %118 = bitcast i8* %add.ptr.i699.us.1 to i16*
  %arrayidx2.i.us.1 = getelementptr inbounds i16, i16* %118, i64 %indvars.iv1851
  store i16 0, i16* %arrayidx2.i.us.1, align 2, !tbaa !29
  %mul.i717.us.1 = mul i64 %113, %indvars.iv.next1849
  %add.ptr.i718.us.1 = getelementptr inbounds i8, i8* %109, i64 %mul.i717.us.1
  %119 = bitcast i8* %add.ptr.i718.us.1 to i16*
  %arrayidx2.i720.us.1 = getelementptr inbounds i16, i16* %119, i64 %indvars.iv1851
  store i16 0, i16* %arrayidx2.i720.us.1, align 2, !tbaa !29
  %indvars.iv.next1849.1 = or i64 %indvars.iv1848, 2
  %mul.i.us.2 = mul i64 %112, %indvars.iv.next1849.1
  %add.ptr.i699.us.2 = getelementptr inbounds i8, i8* %108, i64 %mul.i.us.2
  %120 = bitcast i8* %add.ptr.i699.us.2 to i16*
  %arrayidx2.i.us.2 = getelementptr inbounds i16, i16* %120, i64 %indvars.iv1851
  store i16 0, i16* %arrayidx2.i.us.2, align 2, !tbaa !29
  %mul.i717.us.2 = mul i64 %113, %indvars.iv.next1849.1
  %add.ptr.i718.us.2 = getelementptr inbounds i8, i8* %109, i64 %mul.i717.us.2
  %121 = bitcast i8* %add.ptr.i718.us.2 to i16*
  %arrayidx2.i720.us.2 = getelementptr inbounds i16, i16* %121, i64 %indvars.iv1851
  store i16 0, i16* %arrayidx2.i720.us.2, align 2, !tbaa !29
  %indvars.iv.next1849.2 = or i64 %indvars.iv1848, 3
  %mul.i.us.3 = mul i64 %112, %indvars.iv.next1849.2
  %add.ptr.i699.us.3 = getelementptr inbounds i8, i8* %108, i64 %mul.i.us.3
  %122 = bitcast i8* %add.ptr.i699.us.3 to i16*
  %arrayidx2.i.us.3 = getelementptr inbounds i16, i16* %122, i64 %indvars.iv1851
  store i16 0, i16* %arrayidx2.i.us.3, align 2, !tbaa !29
  %mul.i717.us.3 = mul i64 %113, %indvars.iv.next1849.2
  %add.ptr.i718.us.3 = getelementptr inbounds i8, i8* %109, i64 %mul.i717.us.3
  %123 = bitcast i8* %add.ptr.i718.us.3 to i16*
  %arrayidx2.i720.us.3 = getelementptr inbounds i16, i16* %123, i64 %indvars.iv1851
  store i16 0, i16* %arrayidx2.i720.us.3, align 2, !tbaa !29
  %indvars.iv.next1849.3 = add nuw nsw i64 %indvars.iv1848, 4
  %niter.nsub.3 = add i64 %niter, -4
  %niter.ncmp.3 = icmp eq i64 %niter.nsub.3, 0
  br i1 %niter.ncmp.3, label %for.cond34.for.cond.cleanup37_crit_edge.us.unr-lcssa, label %for.body39.us

for.body39.lr.ph.us:                              ; preds = %pfor.detach.us
  br i1 %115, label %for.cond34.for.cond.cleanup37_crit_edge.us.unr-lcssa, label %for.body39.lr.ph.us.new

for.body39.lr.ph.us.new:                          ; preds = %for.body39.lr.ph.us
  br label %for.body39.us

for.cond34.for.cond.cleanup37_crit_edge.us.unr-lcssa: ; preds = %for.body39.lr.ph.us, %for.body39.us
  %indvars.iv1848.unr = phi i64 [ 0, %for.body39.lr.ph.us ], [ %indvars.iv.next1849.3, %for.body39.us ]
  br i1 %lcmp.mod, label %for.cond34.for.cond.cleanup37_crit_edge.us, label %for.body39.us.epil.preheader

for.body39.us.epil.preheader:                     ; preds = %for.cond34.for.cond.cleanup37_crit_edge.us.unr-lcssa
  br label %for.body39.us.epil

for.body39.us.epil:                               ; preds = %for.body39.us.epil, %for.body39.us.epil.preheader
  %indvars.iv1848.epil = phi i64 [ %indvars.iv1848.unr, %for.body39.us.epil.preheader ], [ %indvars.iv.next1849.epil, %for.body39.us.epil ]
  %epil.iter = phi i64 [ %xtraiter, %for.body39.us.epil.preheader ], [ %epil.iter.sub, %for.body39.us.epil ]
  %mul.i.us.epil = mul i64 %112, %indvars.iv1848.epil
  %add.ptr.i699.us.epil = getelementptr inbounds i8, i8* %108, i64 %mul.i.us.epil
  %124 = bitcast i8* %add.ptr.i699.us.epil to i16*
  %arrayidx2.i.us.epil = getelementptr inbounds i16, i16* %124, i64 %indvars.iv1851
  store i16 0, i16* %arrayidx2.i.us.epil, align 2, !tbaa !29
  %mul.i717.us.epil = mul i64 %113, %indvars.iv1848.epil
  %add.ptr.i718.us.epil = getelementptr inbounds i8, i8* %109, i64 %mul.i717.us.epil
  %125 = bitcast i8* %add.ptr.i718.us.epil to i16*
  %arrayidx2.i720.us.epil = getelementptr inbounds i16, i16* %125, i64 %indvars.iv1851
  store i16 0, i16* %arrayidx2.i720.us.epil, align 2, !tbaa !29
  %indvars.iv.next1849.epil = add nuw nsw i64 %indvars.iv1848.epil, 1
  %epil.iter.sub = add i64 %epil.iter, -1
  %epil.iter.cmp = icmp eq i64 %epil.iter.sub, 0
  br i1 %epil.iter.cmp, label %for.cond34.for.cond.cleanup37_crit_edge.us, label %for.body39.us.epil, !llvm.loop !31

for.cond34.for.cond.cleanup37_crit_edge.us:       ; preds = %for.body39.us.epil, %for.cond34.for.cond.cleanup37_crit_edge.us.unr-lcssa
  reattach within %syncreg203, label %pfor.inc.us

pfor.cond.cleanup:                                ; preds = %pfor.inc.us, %pfor.detach.lr.ph, %invoke.cont29
  sync within %syncreg203, label %sync.continue

lpad25:                                           ; preds = %if.else.i1184, %if.then.i.i1153, %if.else.i, %if.then.i.i1123, %if.end.i694, %invoke.cont24
  %126 = landingpad { i8*, i32 }
          cleanup
  %127 = extractvalue { i8*, i32 } %126, 0
  %128 = extractvalue { i8*, i32 } %126, 1
  br label %ehcleanup360

sync.continue:                                    ; preds = %pfor.cond.cleanup
  %cmp621816 = icmp sgt i32 %106, 6
  br i1 %cmp621816, label %pfor.detach64.lr.ph, label %pfor.cond.cleanup63

pfor.detach64.lr.ph:                              ; preds = %sync.continue
  %sub57 = add nsw i32 %106, -6
  %cmp118 = icmp eq i64 %indvars.iv1884, 0
  %wide.trip.count1872 = zext i32 %sub57 to i64
  br label %pfor.detach64

pfor.cond.cleanup63:                              ; preds = %pfor.inc187, %sync.continue
  sync within %syncreg203, label %sync.continue196

pfor.detach64:                                    ; preds = %pfor.inc187, %pfor.detach64.lr.ph
  %indvars.iv1869 = phi i64 [ 0, %pfor.detach64.lr.ph ], [ %indvars.iv.next1870, %pfor.inc187 ]
  %129 = add nuw nsw i64 %indvars.iv1869, 3
  detach within %syncreg203, label %pfor.body69, label %pfor.inc187

pfor.body69:                                      ; preds = %pfor.detach64
  %130 = load i32, i32* %rows, align 8, !tbaa !22
  %cmp741813 = icmp sgt i32 %130, 6
  br i1 %cmp741813, label %for.body77.lr.ph, label %for.cond.cleanup75

for.body77.lr.ph:                                 ; preds = %pfor.body69
  %sext = shl i64 %129, 32
  %131 = ashr exact i64 %sext, 32
  br label %for.body77

for.cond.cleanup75:                               ; preds = %for.inc182, %pfor.body69
  reattach within %syncreg203, label %pfor.inc187

for.body77:                                       ; preds = %for.inc182, %for.body77.lr.ph
  %indvars.iv1867 = phi i64 [ 3, %for.body77.lr.ph ], [ %indvars.iv.next1868, %for.inc182 ]
  %132 = load i8*, i8** %data.i735, align 8, !tbaa !33
  %133 = load i64*, i64** %p.i736, align 8, !tbaa !34
  %134 = load i64, i64* %133, align 8, !tbaa !25
  %mul.i738 = mul i64 %134, %indvars.iv1867
  %add.ptr.i739 = getelementptr inbounds i8, i8* %132, i64 %mul.i738
  %arrayidx2.i741 = getelementptr inbounds i8, i8* %add.ptr.i739, i64 %129
  %135 = load i8, i8* %arrayidx2.i741, align 1, !tbaa !35
  %cmp84 = icmp eq i8 %135, 0
  br i1 %cmp84, label %if.then85, label %if.end169

if.then85:                                        ; preds = %for.body77
  %136 = add nsw i64 %indvars.iv1867, -2
  %137 = add nsw i64 %indvars.iv1867, -1
  %mul.i872.us = mul i64 %134, %136
  %add.ptr.i873.us = getelementptr inbounds i8, i8* %132, i64 %mul.i872.us
  %mul.i872.us.1 = mul i64 %134, %137
  %add.ptr.i873.us.1 = getelementptr inbounds i8, i8* %132, i64 %mul.i872.us.1
  %138 = add nuw nsw i64 %indvars.iv1867, 1
  %139 = add nuw nsw i64 %indvars.iv1867, 2
  %mul.i872.us.3 = mul i64 %134, %138
  %add.ptr.i873.us.3 = getelementptr inbounds i8, i8* %132, i64 %mul.i872.us.3
  %mul.i872.us.4 = mul i64 %134, %139
  %add.ptr.i873.us.4 = getelementptr inbounds i8, i8* %132, i64 %mul.i872.us.4
  br label %for.body89

for.body89:                                       ; preds = %for.cond.cleanup92, %if.then85
  %indvars.iv1863 = phi i64 [ -2, %if.then85 ], [ %indvars.iv.next1864, %for.cond.cleanup92 ]
  %140 = add nsw i64 %indvars.iv1863, %131
  br i1 %cmp118, label %for.body89.split.us, label %for.body93.preheader

for.body93.preheader:                             ; preds = %for.body89
  br label %for.body93

for.body89.split.us:                              ; preds = %for.body89
  %141 = load i8*, i8** %data.i756, align 16, !tbaa !33
  %142 = load i64*, i64** %p.i757, align 8, !tbaa !34
  %143 = load i64, i64* %142, align 8, !tbaa !25
  %mul.i759.us = mul i64 %143, %136
  %add.ptr.i760.us = getelementptr inbounds i8, i8* %141, i64 %mul.i759.us
  %arrayidx2.i762.us = getelementptr inbounds i8, i8* %add.ptr.i760.us, i64 %140
  %144 = load i8, i8* %arrayidx2.i762.us, align 1, !tbaa !35
  %cmp100.us = icmp eq i8 %144, 0
  br i1 %cmp100.us, label %if.end141.us, label %if.then101.us

if.then101.us:                                    ; preds = %for.body89.split.us
  %conv106.us = zext i8 %144 to i16
  %145 = load i8*, i8** %data.i697, align 8, !tbaa !33
  %146 = load i64*, i64** %p.i3.i638, align 8, !tbaa !34
  %147 = load i64, i64* %146, align 8, !tbaa !25
  %mul.i801.us = mul i64 %147, %indvars.iv1867
  %add.ptr.i802.us = getelementptr inbounds i8, i8* %145, i64 %mul.i801.us
  %148 = bitcast i8* %add.ptr.i802.us to i16*
  %arrayidx2.i804.us = getelementptr inbounds i16, i16* %148, i64 %129
  %149 = load i16, i16* %arrayidx2.i804.us, align 2, !tbaa !29
  %add110.us = add i16 %149, %conv106.us
  store i16 %add110.us, i16* %arrayidx2.i804.us, align 2, !tbaa !29
  %150 = load i8*, i8** %data.i714, align 8, !tbaa !33
  %151 = load i64*, i64** %p.i3.i658, align 8, !tbaa !34
  %152 = load i64, i64* %151, align 8, !tbaa !25
  %mul.i822.us = mul i64 %152, %indvars.iv1867
  %add.ptr.i823.us = getelementptr inbounds i8, i8* %150, i64 %mul.i822.us
  %153 = bitcast i8* %add.ptr.i823.us to i16*
  %arrayidx2.i825.us = getelementptr inbounds i16, i16* %153, i64 %129
  %154 = load i16, i16* %arrayidx2.i825.us, align 2, !tbaa !29
  %add115.us = add i16 %154, 1
  store i16 %add115.us, i16* %arrayidx2.i825.us, align 2, !tbaa !29
  br label %if.end141.us

if.end141.us:                                     ; preds = %if.then101.us, %for.body89.split.us
  %155 = load i8, i8* %arrayidx2.i741, align 1, !tbaa !35
  %cmp145.us = icmp eq i8 %155, 0
  br i1 %cmp145.us, label %for.inc163.us, label %if.then146.us

if.then146.us:                                    ; preds = %if.end141.us
  %arrayidx2.i875.us = getelementptr inbounds i8, i8* %add.ptr.i873.us, i64 %140
  %156 = load i8, i8* %arrayidx2.i875.us, align 1, !tbaa !35
  %conv151.us = zext i8 %156 to i16
  %157 = load i8*, i8** %data.i697, align 8, !tbaa !33
  %158 = load i64*, i64** %p.i3.i638, align 8, !tbaa !34
  %159 = load i64, i64* %158, align 8, !tbaa !25
  %mul.i884.us = mul i64 %159, %indvars.iv1867
  %add.ptr.i885.us = getelementptr inbounds i8, i8* %157, i64 %mul.i884.us
  %160 = bitcast i8* %add.ptr.i885.us to i16*
  %arrayidx2.i887.us = getelementptr inbounds i16, i16* %160, i64 %129
  %161 = load i16, i16* %arrayidx2.i887.us, align 2, !tbaa !29
  %add155.us = add i16 %161, %conv151.us
  store i16 %add155.us, i16* %arrayidx2.i887.us, align 2, !tbaa !29
  %162 = load i8*, i8** %data.i714, align 8, !tbaa !33
  %163 = load i64*, i64** %p.i3.i658, align 8, !tbaa !34
  %164 = load i64, i64* %163, align 8, !tbaa !25
  %mul.i891.us = mul i64 %164, %indvars.iv1867
  %add.ptr.i892.us = getelementptr inbounds i8, i8* %162, i64 %mul.i891.us
  %165 = bitcast i8* %add.ptr.i892.us to i16*
  %arrayidx2.i894.us = getelementptr inbounds i16, i16* %165, i64 %129
  %166 = load i16, i16* %arrayidx2.i894.us, align 2, !tbaa !29
  %add160.us = add i16 %166, 1
  store i16 %add160.us, i16* %arrayidx2.i894.us, align 2, !tbaa !29
  br label %for.inc163.us

for.inc163.us:                                    ; preds = %if.then146.us, %if.end141.us
  %mul.i759.us.1 = mul i64 %143, %137
  %add.ptr.i760.us.1 = getelementptr inbounds i8, i8* %141, i64 %mul.i759.us.1
  %arrayidx2.i762.us.1 = getelementptr inbounds i8, i8* %add.ptr.i760.us.1, i64 %140
  %167 = load i8, i8* %arrayidx2.i762.us.1, align 1, !tbaa !35
  %cmp100.us.1 = icmp eq i8 %167, 0
  br i1 %cmp100.us.1, label %if.end141.us.1, label %if.then101.us.1

for.cond.cleanup92:                               ; preds = %if.then146.us.4, %if.end141.us.4, %for.inc163
  %indvars.iv.next1864 = add nsw i64 %indvars.iv1863, 1
  %exitcond1866 = icmp eq i64 %indvars.iv.next1864, 3
  br i1 %exitcond1866, label %if.end169, label %for.body89

for.body93:                                       ; preds = %for.inc163, %for.body93.preheader
  %indvars.iv1855 = phi i64 [ %indvars.iv.next1856, %for.inc163 ], [ -2, %for.body93.preheader ]
  %168 = add nsw i64 %indvars.iv1855, %indvars.iv1867
  %169 = load i8*, i8** %data.i756, align 16, !tbaa !33
  %170 = load i64*, i64** %p.i757, align 8, !tbaa !34
  %171 = load i64, i64* %170, align 8, !tbaa !25
  %mul.i759 = mul i64 %171, %168
  %add.ptr.i760 = getelementptr inbounds i8, i8* %169, i64 %mul.i759
  %arrayidx2.i762 = getelementptr inbounds i8, i8* %add.ptr.i760, i64 %140
  %172 = load i8, i8* %arrayidx2.i762, align 1, !tbaa !35
  %cmp100 = icmp eq i8 %172, 0
  br i1 %cmp100, label %land.lhs.true, label %if.then101

if.then101:                                       ; preds = %for.body93
  %conv106 = zext i8 %172 to i16
  %173 = load i8*, i8** %data.i697, align 8, !tbaa !33
  %174 = load i64*, i64** %p.i3.i638, align 8, !tbaa !34
  %175 = load i64, i64* %174, align 8, !tbaa !25
  %mul.i801 = mul i64 %175, %indvars.iv1867
  %add.ptr.i802 = getelementptr inbounds i8, i8* %173, i64 %mul.i801
  %176 = bitcast i8* %add.ptr.i802 to i16*
  %arrayidx2.i804 = getelementptr inbounds i16, i16* %176, i64 %129
  %177 = load i16, i16* %arrayidx2.i804, align 2, !tbaa !29
  %add110 = add i16 %177, %conv106
  store i16 %add110, i16* %arrayidx2.i804, align 2, !tbaa !29
  %178 = load i8*, i8** %data.i714, align 8, !tbaa !33
  %179 = load i64*, i64** %p.i3.i658, align 8, !tbaa !34
  %180 = load i64, i64* %179, align 8, !tbaa !25
  %mul.i822 = mul i64 %180, %indvars.iv1867
  %add.ptr.i823 = getelementptr inbounds i8, i8* %178, i64 %mul.i822
  %181 = bitcast i8* %add.ptr.i823 to i16*
  %arrayidx2.i825 = getelementptr inbounds i16, i16* %181, i64 %129
  %182 = load i16, i16* %arrayidx2.i825, align 2, !tbaa !29
  %add115 = add i16 %182, 1
  store i16 %add115, i16* %arrayidx2.i825, align 2, !tbaa !29
  br label %land.lhs.true

land.lhs.true:                                    ; preds = %if.then101, %for.body93
  %183 = load i8*, i8** %data.i826, align 8, !tbaa !33
  %184 = load i64*, i64** %p.i3.i, align 8, !tbaa !34
  %185 = load i64, i64* %184, align 8, !tbaa !25
  %mul.i829 = mul i64 %185, %168
  %add.ptr.i830 = getelementptr inbounds i8, i8* %183, i64 %mul.i829
  %arrayidx2.i832 = getelementptr inbounds i8, i8* %add.ptr.i830, i64 %140
  %186 = load i8, i8* %arrayidx2.i832, align 1, !tbaa !35
  %cmp124 = icmp eq i8 %186, 0
  br i1 %cmp124, label %if.end141, label %if.then125

if.then125:                                       ; preds = %land.lhs.true
  %conv130 = zext i8 %186 to i16
  %187 = load i8*, i8** %data.i697, align 8, !tbaa !33
  %188 = load i64*, i64** %p.i3.i638, align 8, !tbaa !34
  %189 = load i64, i64* %188, align 8, !tbaa !25
  %mul.i845 = mul i64 %189, %indvars.iv1867
  %add.ptr.i846 = getelementptr inbounds i8, i8* %187, i64 %mul.i845
  %190 = bitcast i8* %add.ptr.i846 to i16*
  %arrayidx2.i848 = getelementptr inbounds i16, i16* %190, i64 %129
  %191 = load i16, i16* %arrayidx2.i848, align 2, !tbaa !29
  %add134 = add i16 %191, %conv130
  store i16 %add134, i16* %arrayidx2.i848, align 2, !tbaa !29
  %192 = load i8*, i8** %data.i714, align 8, !tbaa !33
  %193 = load i64*, i64** %p.i3.i658, align 8, !tbaa !34
  %194 = load i64, i64* %193, align 8, !tbaa !25
  %mul.i857 = mul i64 %194, %indvars.iv1867
  %add.ptr.i858 = getelementptr inbounds i8, i8* %192, i64 %mul.i857
  %195 = bitcast i8* %add.ptr.i858 to i16*
  %arrayidx2.i860 = getelementptr inbounds i16, i16* %195, i64 %129
  %196 = load i16, i16* %arrayidx2.i860, align 2, !tbaa !29
  %add139 = add i16 %196, 1
  store i16 %add139, i16* %arrayidx2.i860, align 2, !tbaa !29
  br label %if.end141

if.end141:                                        ; preds = %if.then125, %land.lhs.true
  %197 = load i8, i8* %arrayidx2.i741, align 1, !tbaa !35
  %cmp145 = icmp eq i8 %197, 0
  br i1 %cmp145, label %for.inc163, label %if.then146

if.then146:                                       ; preds = %if.end141
  %mul.i872 = mul i64 %134, %168
  %add.ptr.i873 = getelementptr inbounds i8, i8* %132, i64 %mul.i872
  %arrayidx2.i875 = getelementptr inbounds i8, i8* %add.ptr.i873, i64 %140
  %198 = load i8, i8* %arrayidx2.i875, align 1, !tbaa !35
  %conv151 = zext i8 %198 to i16
  %199 = load i8*, i8** %data.i697, align 8, !tbaa !33
  %200 = load i64*, i64** %p.i3.i638, align 8, !tbaa !34
  %201 = load i64, i64* %200, align 8, !tbaa !25
  %mul.i884 = mul i64 %201, %indvars.iv1867
  %add.ptr.i885 = getelementptr inbounds i8, i8* %199, i64 %mul.i884
  %202 = bitcast i8* %add.ptr.i885 to i16*
  %arrayidx2.i887 = getelementptr inbounds i16, i16* %202, i64 %129
  %203 = load i16, i16* %arrayidx2.i887, align 2, !tbaa !29
  %add155 = add i16 %203, %conv151
  store i16 %add155, i16* %arrayidx2.i887, align 2, !tbaa !29
  %204 = load i8*, i8** %data.i714, align 8, !tbaa !33
  %205 = load i64*, i64** %p.i3.i658, align 8, !tbaa !34
  %206 = load i64, i64* %205, align 8, !tbaa !25
  %mul.i891 = mul i64 %206, %indvars.iv1867
  %add.ptr.i892 = getelementptr inbounds i8, i8* %204, i64 %mul.i891
  %207 = bitcast i8* %add.ptr.i892 to i16*
  %arrayidx2.i894 = getelementptr inbounds i16, i16* %207, i64 %129
  %208 = load i16, i16* %arrayidx2.i894, align 2, !tbaa !29
  %add160 = add i16 %208, 1
  store i16 %add160, i16* %arrayidx2.i894, align 2, !tbaa !29
  br label %for.inc163

for.inc163:                                       ; preds = %if.then146, %if.end141
  %indvars.iv.next1856 = add nsw i64 %indvars.iv1855, 1
  %exitcond1858 = icmp eq i64 %indvars.iv.next1856, 3
  br i1 %exitcond1858, label %for.cond.cleanup92, label %for.body93

if.end169:                                        ; preds = %for.cond.cleanup92, %for.body77
  br i1 %cmp118, label %for.inc182, label %land.lhs.true171

land.lhs.true171:                                 ; preds = %if.end169
  %209 = load i8, i8* %arrayidx2.i741, align 1, !tbaa !35
  %cmp175 = icmp eq i8 %209, 0
  br i1 %cmp175, label %if.then176, label %for.inc182

if.then176:                                       ; preds = %land.lhs.true171
  %210 = load i8*, i8** %data.i826, align 8, !tbaa !33
  %211 = load i64*, i64** %p.i3.i, align 8, !tbaa !34
  %212 = load i64, i64* %211, align 8, !tbaa !25
  %mul.i905 = mul i64 %212, %indvars.iv1867
  %add.ptr.i906 = getelementptr inbounds i8, i8* %210, i64 %mul.i905
  %arrayidx2.i908 = getelementptr inbounds i8, i8* %add.ptr.i906, i64 %129
  %213 = load i8, i8* %arrayidx2.i908, align 1, !tbaa !35
  store i8 %213, i8* %arrayidx2.i741, align 1, !tbaa !35
  br label %for.inc182

for.inc182:                                       ; preds = %if.then176, %land.lhs.true171, %if.end169
  %indvars.iv.next1868 = add nuw nsw i64 %indvars.iv1867, 1
  %214 = load i32, i32* %rows, align 8, !tbaa !22
  %sub73 = add nsw i32 %214, -3
  %215 = sext i32 %sub73 to i64
  %cmp74 = icmp slt i64 %indvars.iv.next1868, %215
  br i1 %cmp74, label %for.body77, label %for.cond.cleanup75

pfor.inc187:                                      ; preds = %for.cond.cleanup75, %pfor.detach64
  %indvars.iv.next1870 = add nuw nsw i64 %indvars.iv1869, 1
  %exitcond1873 = icmp eq i64 %indvars.iv.next1870, %wide.trip.count1872
  br i1 %exitcond1873, label %pfor.cond.cleanup63, label %pfor.detach64, !llvm.loop !36

sync.continue196:                                 ; preds = %pfor.cond.cleanup63
  %216 = load i32, i32* %cols, align 4, !tbaa !23
  %cmp2141820 = icmp sgt i32 %216, 6
  br i1 %cmp2141820, label %pfor.detach216.lr.ph, label %pfor.cond.cleanup215

pfor.detach216.lr.ph:                             ; preds = %sync.continue196
  %sub209 = add nsw i32 %216, -6
  %wide.trip.count1882 = zext i32 %sub209 to i64
  br label %pfor.detach216

pfor.cond.cleanup215:                             ; preds = %pfor.inc284, %sync.continue196
  sync within %syncreg203, label %sync.continue293

pfor.detach216:                                   ; preds = %pfor.inc284, %pfor.detach216.lr.ph
  %indvars.iv1879 = phi i64 [ 0, %pfor.detach216.lr.ph ], [ %indvars.iv.next1880, %pfor.inc284 ]
  %217 = add nuw nsw i64 %indvars.iv1879, 3
  detach within %syncreg203, label %pfor.body221, label %pfor.inc284

pfor.body221:                                     ; preds = %pfor.detach216
  %syncreg222 = call token @llvm.syncregion.start()
  %218 = load i32, i32* %rows, align 8, !tbaa !22
  %cmp2331818 = icmp sgt i32 %218, 6
  br i1 %cmp2331818, label %pfor.detach236.lr.ph, label %pfor.cond.cleanup234

pfor.detach236.lr.ph:                             ; preds = %pfor.body221
  %sub228 = add nsw i32 %218, -6
  %wide.trip.count1877 = zext i32 %sub228 to i64
  br label %pfor.detach236

pfor.cond.cleanup234:                             ; preds = %pfor.inc263, %pfor.body221
  sync within %syncreg222, label %sync.continue274

pfor.detach236:                                   ; preds = %pfor.inc263, %pfor.detach236.lr.ph
  %indvars.iv1874 = phi i64 [ 0, %pfor.detach236.lr.ph ], [ %indvars.iv.next1875, %pfor.inc263 ]
  detach within %syncreg222, label %pfor.body241, label %pfor.inc263

pfor.body241:                                     ; preds = %pfor.detach236
  %219 = add nuw nsw i64 %indvars.iv1874, 3
  %220 = load i8*, i8** %data.i714, align 8, !tbaa !33
  %221 = load i64*, i64** %p.i3.i658, align 8, !tbaa !34
  %222 = load i64, i64* %221, align 8, !tbaa !25
  %mul.i952 = mul i64 %222, %219
  %add.ptr.i953 = getelementptr inbounds i8, i8* %220, i64 %mul.i952
  %223 = bitcast i8* %add.ptr.i953 to i16*
  %arrayidx2.i955 = getelementptr inbounds i16, i16* %223, i64 %217
  %224 = load i16, i16* %arrayidx2.i955, align 2, !tbaa !29
  %cmp248 = icmp eq i16 %224, 0
  br i1 %cmp248, label %pfor.preattach261, label %if.then249

if.then249:                                       ; preds = %pfor.body241
  %225 = load i8*, i8** %data.i697, align 8, !tbaa !33
  %226 = load i64*, i64** %p.i3.i638, align 8, !tbaa !34
  %227 = load i64, i64* %226, align 8, !tbaa !25
  %mul.i959 = mul i64 %227, %219
  %add.ptr.i960 = getelementptr inbounds i8, i8* %225, i64 %mul.i959
  %228 = bitcast i8* %add.ptr.i960 to i16*
  %arrayidx2.i962 = getelementptr inbounds i16, i16* %228, i64 %217
  %229 = load i16, i16* %arrayidx2.i962, align 2, !tbaa !29
  %230 = udiv i16 %229, %224
  %conv257 = trunc i16 %230 to i8
  %231 = load i8*, i8** %data.i735, align 8, !tbaa !33
  %232 = load i64*, i64** %p.i736, align 8, !tbaa !34
  %233 = load i64, i64* %232, align 8, !tbaa !25
  %mul.i998 = mul i64 %233, %219
  %add.ptr.i999 = getelementptr inbounds i8, i8* %231, i64 %mul.i998
  %arrayidx2.i1001 = getelementptr inbounds i8, i8* %add.ptr.i999, i64 %217
  store i8 %conv257, i8* %arrayidx2.i1001, align 1, !tbaa !35
  br label %pfor.preattach261

pfor.preattach261:                                ; preds = %if.then249, %pfor.body241
  reattach within %syncreg222, label %pfor.inc263

pfor.inc263:                                      ; preds = %pfor.preattach261, %pfor.detach236
  %indvars.iv.next1875 = add nuw nsw i64 %indvars.iv1874, 1
  %exitcond1878 = icmp eq i64 %indvars.iv.next1875, %wide.trip.count1877
  br i1 %exitcond1878, label %pfor.cond.cleanup234, label %pfor.detach236, !llvm.loop !37

sync.continue274:                                 ; preds = %pfor.cond.cleanup234
  reattach within %syncreg203, label %pfor.inc284

pfor.inc284:                                      ; preds = %sync.continue274, %pfor.detach216
  %indvars.iv.next1880 = add nuw nsw i64 %indvars.iv1879, 1
  %exitcond1883 = icmp eq i64 %indvars.iv.next1880, %wide.trip.count1882
  br i1 %exitcond1883, label %pfor.cond.cleanup215, label %pfor.detach216, !llvm.loop !38

sync.continue293:                                 ; preds = %pfor.cond.cleanup215
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %43)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %44)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %45)
  invoke void @_ZNSsC1ERKSs(%"class.std::basic_string"* nonnull %ref.tmp302, %"class.std::basic_string"* nonnull dereferenceable(8) %filename_prefix)
          to label %.noexc1019 unwind label %lpad303

.noexc1019:                                       ; preds = %sync.continue293
  %call2.i2.i1004 = invoke dereferenceable(8) %"class.std::basic_string"* @_ZNSs6appendEPKcm(%"class.std::basic_string"* nonnull %ref.tmp302, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.4, i64 0, i64 0), i64 1)
          to label %invoke.cont304 unwind label %lpad.i1008

lpad.i1008:                                       ; preds = %.noexc1019
  %234 = landingpad { i8*, i32 }
          cleanup
  %235 = load i8*, i8** %_M_p.i.i.i.i1024, align 8, !tbaa !9, !alias.scope !39
  %arrayidx.i.i.i1006 = getelementptr inbounds i8, i8* %235, i64 -24
  %236 = bitcast i8* %arrayidx.i.i.i1006 to %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"*
  %237 = getelementptr inbounds %"class.std::allocator", %"class.std::allocator"* %ref.tmp.i.i1002, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %237), !noalias !39
  %cmp.i.i.i1007 = icmp eq i8* %arrayidx.i.i.i1006, bitcast ([0 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE to i8*)
  br i1 %cmp.i.i.i1007, label %_ZNSsD2Ev.exit.i1018, label %if.then.i.i.i1010, !prof !12

if.then.i.i.i1010:                                ; preds = %lpad.i1008
  %_M_refcount.i.i.i1009 = getelementptr inbounds i8, i8* %235, i64 -8
  %238 = bitcast i8* %_M_refcount.i.i.i1009 to i32*
  br i1 icmp ne (i8* bitcast (i32 (i32*, void (i8*)*)* @__pthread_key_create to i8*), i8* null), label %if.then.i.i.i.i1011, label %if.else.i.i.i.i1013

if.then.i.i.i.i1011:                              ; preds = %if.then.i.i.i1010
  %239 = atomicrmw volatile add i32* %238, i32 -1 acq_rel
  br label %invoke.cont.i.i.i1016

if.else.i.i.i.i1013:                              ; preds = %if.then.i.i.i1010
  %240 = load i32, i32* %238, align 4, !tbaa !13
  %add.i.i.i.i.i1012 = add nsw i32 %240, -1
  store i32 %add.i.i.i.i.i1012, i32* %238, align 4, !tbaa !13
  br label %invoke.cont.i.i.i1016

invoke.cont.i.i.i1016:                            ; preds = %if.else.i.i.i.i1013, %if.then.i.i.i.i1011
  %retval.0.i.i.i.i1014 = phi i32 [ %239, %if.then.i.i.i.i1011 ], [ %240, %if.else.i.i.i.i1013 ]
  %cmp3.i.i.i1015 = icmp slt i32 %retval.0.i.i.i.i1014, 1
  br i1 %cmp3.i.i.i1015, label %if.then4.i.i.i1017, label %_ZNSsD2Ev.exit.i1018

if.then4.i.i.i1017:                               ; preds = %invoke.cont.i.i.i1016
  call void @_ZNSs4_Rep10_M_destroyERKSaIcE(%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"* nonnull %236, %"class.std::allocator"* nonnull dereferenceable(1) %ref.tmp.i.i1002)
  br label %_ZNSsD2Ev.exit.i1018

_ZNSsD2Ev.exit.i1018:                             ; preds = %if.then4.i.i.i1017, %invoke.cont.i.i.i1016, %lpad.i1008
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %237), !noalias !39
  br label %lpad303.body

invoke.cont304:                                   ; preds = %.noexc1019
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %46)
  %real_section_id = getelementptr inbounds %"class.tfk::Section", %"class.tfk::Section"* %96, i64 0, i32 1
  %241 = load i32, i32* %real_section_id, align 4, !tbaa !42
  invoke void (%"class.std::basic_string"*, i32 (i8*, i64, i8*, %struct.__va_list_tag*)*, i64, i8*, ...) @_ZN9__gnu_cxx12__to_xstringISscEET_PFiPT0_mPKS2_P13__va_list_tagEmS5_z(%"class.std::basic_string"* nonnull sret %ref.tmp305, i32 (i8*, i64, i8*, %struct.__va_list_tag*)* nonnull @vsnprintf, i64 16, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.16, i64 0, i64 0), i32 %241)
          to label %invoke.cont307 unwind label %lpad306

invoke.cont307:                                   ; preds = %invoke.cont304
  %242 = load i8*, i8** %_M_p.i.i.i.i1024, align 8, !tbaa !9, !noalias !57
  %arrayidx.i.i.i1025 = getelementptr inbounds i8, i8* %242, i64 -24
  %_M_length.i.i1026 = bitcast i8* %arrayidx.i.i.i1025 to i64*
  %243 = load i64, i64* %_M_length.i.i1026, align 8, !tbaa !60, !noalias !57
  %244 = load i8*, i8** %_M_p.i.i.i17.i1027, align 8, !tbaa !9, !noalias !57
  %arrayidx.i.i18.i1028 = getelementptr inbounds i8, i8* %244, i64 -24
  %_M_length.i19.i1029 = bitcast i8* %arrayidx.i.i18.i1028 to i64*
  %245 = load i64, i64* %_M_length.i19.i1029, align 8, !tbaa !60, !noalias !57
  %add.i1030 = add i64 %245, %243
  %_M_capacity.i22.i1031 = getelementptr inbounds i8, i8* %242, i64 -16
  %246 = bitcast i8* %_M_capacity.i22.i1031 to i64*
  %247 = load i64, i64* %246, align 8, !tbaa !62, !noalias !57
  %cmp.i1032 = icmp ugt i64 %add.i1030, %247
  br i1 %cmp.i1032, label %land.rhs.i1035, label %cond.false.i1037

land.rhs.i1035:                                   ; preds = %invoke.cont307
  %_M_capacity.i.i1033 = getelementptr inbounds i8, i8* %244, i64 -16
  %248 = bitcast i8* %_M_capacity.i.i1033 to i64*
  %249 = load i64, i64* %248, align 8, !tbaa !62, !noalias !57
  %cmp4.i1034 = icmp ugt i64 %add.i1030, %249
  br i1 %cmp4.i1034, label %cond.false.i1037, label %cond.true.i1036

cond.true.i1036:                                  ; preds = %land.rhs.i1035
  %call4.i.i.i1041 = invoke dereferenceable(8) %"class.std::basic_string"* @_ZNSs6insertEmPKcm(%"class.std::basic_string"* nonnull %ref.tmp305, i64 0, i8* %242, i64 %243)
          to label %invoke.cont309 unwind label %lpad308

cond.false.i1037:                                 ; preds = %land.rhs.i1035, %invoke.cont307
  %call7.i1043 = invoke dereferenceable(8) %"class.std::basic_string"* @_ZNSs6appendERKSs(%"class.std::basic_string"* nonnull %ref.tmp302, %"class.std::basic_string"* nonnull dereferenceable(8) %ref.tmp305)
          to label %invoke.cont309 unwind label %lpad308

invoke.cont309:                                   ; preds = %cond.false.i1037, %cond.true.i1036
  %cond-lvalue.i1038 = phi %"class.std::basic_string"* [ %call4.i.i.i1041, %cond.true.i1036 ], [ %call7.i1043, %cond.false.i1037 ]
  %250 = bitcast %"class.std::basic_string"* %cond-lvalue.i1038 to i64*
  %251 = load i64, i64* %250, align 8, !tbaa !63, !noalias !57
  store i64 %251, i64* %47, align 8, !tbaa !63, !alias.scope !57
  %_M_p.i.i.i1039 = getelementptr inbounds %"class.std::basic_string", %"class.std::basic_string"* %cond-lvalue.i1038, i64 0, i32 0, i32 0
  store i8* bitcast (i64* getelementptr inbounds ([0 x i64], [0 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE, i64 0, i64 3) to i8*), i8** %_M_p.i.i.i1039, align 8, !tbaa !9, !noalias !57
  %call2.i.i1047 = invoke dereferenceable(8) %"class.std::basic_string"* @_ZNSs6appendEPKcm(%"class.std::basic_string"* nonnull %ref.tmp301, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.5, i64 0, i64 0), i64 4)
          to label %invoke.cont311 unwind label %lpad310

invoke.cont311:                                   ; preds = %invoke.cont309
  %252 = getelementptr inbounds %"class.std::basic_string", %"class.std::basic_string"* %call2.i.i1047, i64 0, i32 0, i32 0
  %253 = load i8*, i8** %252, align 8, !tbaa !63, !noalias !64
  store i8* bitcast (i64* getelementptr inbounds ([0 x i64], [0 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE, i64 0, i64 3) to i8*), i8** %252, align 8, !tbaa !9, !noalias !64
  call void @llvm.memset.p0i8.i64(i8* nonnull %43, i8 0, i64 16, i32 8, i1 false)
  %arrayidx.i.i.i.i = getelementptr inbounds i8, i8* %253, i64 -24
  %_M_length.i.i.i = bitcast i8* %arrayidx.i.i.i.i to i64*
  %254 = load i64, i64* %_M_length.i.i.i, align 8, !tbaa !60
  %cmp.i.i1049 = icmp eq i64 %254, 0
  br i1 %cmp.i.i1049, label %invoke.cont316, label %if.then.i1050

if.then.i1050:                                    ; preds = %invoke.cont311
  %call3.i1052 = invoke i8* @_ZN2cv6String8allocateEm(%"class.cv::String"* nonnull %ref.tmp, i64 %254)
          to label %call3.i.noexc unwind label %lpad312

call3.i.noexc:                                    ; preds = %if.then.i1050
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %call3.i1052, i8* %253, i64 %254, i32 1, i1 false)
  br label %invoke.cont316

invoke.cont316:                                   ; preds = %call3.i.noexc, %invoke.cont311
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %48)
  store i32 0, i32* %width.i.i, align 8, !tbaa !67
  store i32 0, i32* %height.i.i, align 4, !tbaa !69
  store i32 16842752, i32* %flags.i.i1053, align 8, !tbaa !70
  store %"class.cv::Mat"* %img, %"class.cv::Mat"** %49, align 8, !tbaa !72
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %50)
  call void @llvm.memset.p0i8.i64(i8* nonnull %50, i8 0, i64 24, i32 8, i1 false)
  %call320 = invoke zeroext i1 @_ZN2cv7imwriteERKNS_6StringERKNS_11_InputArrayERKSt6vectorIiSaIiEE(%"class.cv::String"* nonnull dereferenceable(16) %ref.tmp, %"class.cv::_InputArray"* nonnull dereferenceable(24) %ref.tmp314, %"class.std::vector.95"* nonnull dereferenceable(24) %ref.tmp317)
          to label %invoke.cont319 unwind label %lpad318

invoke.cont319:                                   ; preds = %invoke.cont316
  %255 = load i32*, i32** %_M_start.i.i, align 8, !tbaa !73
  %tobool.i.i.i = icmp eq i32* %255, null
  br i1 %tobool.i.i.i, label %_ZNSt6vectorIiSaIiEED2Ev.exit, label %if.then.i.i.i1056

if.then.i.i.i1056:                                ; preds = %invoke.cont319
  %256 = bitcast i32* %255 to i8*
  call void @_ZdlPv(i8* %256)
  br label %_ZNSt6vectorIiSaIiEED2Ev.exit

_ZNSt6vectorIiSaIiEED2Ev.exit:                    ; preds = %if.then.i.i.i1056, %invoke.cont319
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %50)
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %48)
  invoke void @_ZN2cv6String10deallocateEv(%"class.cv::String"* nonnull %ref.tmp)
          to label %_ZN2cv6StringD2Ev.exit unwind label %terminate.lpad.i1058

terminate.lpad.i1058:                             ; preds = %_ZNSt6vectorIiSaIiEED2Ev.exit
  %257 = landingpad { i8*, i32 }
          catch i8* null
  %258 = extractvalue { i8*, i32 } %257, 0
  call void @__clang_call_terminate(i8* %258)
  unreachable

_ZN2cv6StringD2Ev.exit:                           ; preds = %_ZNSt6vectorIiSaIiEED2Ev.exit
  %259 = bitcast i8* %arrayidx.i.i.i.i to %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %51)
  %cmp.i.i1062 = icmp eq i8* %arrayidx.i.i.i.i, bitcast ([0 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE to i8*)
  br i1 %cmp.i.i1062, label %_ZNSsD2Ev.exit1072, label %if.then.i.i1064, !prof !12

if.then.i.i1064:                                  ; preds = %_ZN2cv6StringD2Ev.exit
  %_M_refcount.i.i1063 = getelementptr inbounds i8, i8* %253, i64 -8
  %260 = bitcast i8* %_M_refcount.i.i1063 to i32*
  br i1 icmp ne (i8* bitcast (i32 (i32*, void (i8*)*)* @__pthread_key_create to i8*), i8* null), label %if.then.i.i.i1065, label %if.else.i.i.i1067

if.then.i.i.i1065:                                ; preds = %if.then.i.i1064
  %261 = atomicrmw volatile add i32* %260, i32 -1 acq_rel
  br label %invoke.cont.i.i1070

if.else.i.i.i1067:                                ; preds = %if.then.i.i1064
  %262 = load i32, i32* %260, align 4, !tbaa !13
  %add.i.i.i.i1066 = add nsw i32 %262, -1
  store i32 %add.i.i.i.i1066, i32* %260, align 4, !tbaa !13
  br label %invoke.cont.i.i1070

invoke.cont.i.i1070:                              ; preds = %if.else.i.i.i1067, %if.then.i.i.i1065
  %retval.0.i.i.i1068 = phi i32 [ %261, %if.then.i.i.i1065 ], [ %262, %if.else.i.i.i1067 ]
  %cmp3.i.i1069 = icmp slt i32 %retval.0.i.i.i1068, 1
  br i1 %cmp3.i.i1069, label %if.then4.i.i1071, label %_ZNSsD2Ev.exit1072

if.then4.i.i1071:                                 ; preds = %invoke.cont.i.i1070
  call void @_ZNSs4_Rep10_M_destroyERKSaIcE(%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"* nonnull %259, %"class.std::allocator"* nonnull dereferenceable(1) %ref.tmp.i1059)
  br label %_ZNSsD2Ev.exit1072

_ZNSsD2Ev.exit1072:                               ; preds = %if.then4.i.i1071, %invoke.cont.i.i1070, %_ZN2cv6StringD2Ev.exit
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %51)
  %263 = load i8*, i8** %_M_p.i.i.i1074, align 8, !tbaa !9
  %arrayidx.i.i1075 = getelementptr inbounds i8, i8* %263, i64 -24
  %264 = bitcast i8* %arrayidx.i.i1075 to %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %52)
  %cmp.i.i1076 = icmp eq i8* %arrayidx.i.i1075, bitcast ([0 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE to i8*)
  br i1 %cmp.i.i1076, label %_ZNSsD2Ev.exit1086, label %if.then.i.i1078, !prof !12

if.then.i.i1078:                                  ; preds = %_ZNSsD2Ev.exit1072
  %_M_refcount.i.i1077 = getelementptr inbounds i8, i8* %263, i64 -8
  %265 = bitcast i8* %_M_refcount.i.i1077 to i32*
  br i1 icmp ne (i8* bitcast (i32 (i32*, void (i8*)*)* @__pthread_key_create to i8*), i8* null), label %if.then.i.i.i1079, label %if.else.i.i.i1081

if.then.i.i.i1079:                                ; preds = %if.then.i.i1078
  %266 = atomicrmw volatile add i32* %265, i32 -1 acq_rel
  br label %invoke.cont.i.i1084

if.else.i.i.i1081:                                ; preds = %if.then.i.i1078
  %267 = load i32, i32* %265, align 4, !tbaa !13
  %add.i.i.i.i1080 = add nsw i32 %267, -1
  store i32 %add.i.i.i.i1080, i32* %265, align 4, !tbaa !13
  br label %invoke.cont.i.i1084

invoke.cont.i.i1084:                              ; preds = %if.else.i.i.i1081, %if.then.i.i.i1079
  %retval.0.i.i.i1082 = phi i32 [ %266, %if.then.i.i.i1079 ], [ %267, %if.else.i.i.i1081 ]
  %cmp3.i.i1083 = icmp slt i32 %retval.0.i.i.i1082, 1
  br i1 %cmp3.i.i1083, label %if.then4.i.i1085, label %_ZNSsD2Ev.exit1086

if.then4.i.i1085:                                 ; preds = %invoke.cont.i.i1084
  call void @_ZNSs4_Rep10_M_destroyERKSaIcE(%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"* nonnull %264, %"class.std::allocator"* nonnull dereferenceable(1) %ref.tmp.i1073)
  br label %_ZNSsD2Ev.exit1086

_ZNSsD2Ev.exit1086:                               ; preds = %if.then4.i.i1085, %invoke.cont.i.i1084, %_ZNSsD2Ev.exit1072
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %52)
  %268 = load i8*, i8** %_M_p.i.i.i17.i1027, align 8, !tbaa !9
  %arrayidx.i.i1089 = getelementptr inbounds i8, i8* %268, i64 -24
  %269 = bitcast i8* %arrayidx.i.i1089 to %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %53)
  %cmp.i.i1090 = icmp eq i8* %arrayidx.i.i1089, bitcast ([0 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE to i8*)
  br i1 %cmp.i.i1090, label %_ZNSsD2Ev.exit1100, label %if.then.i.i1092, !prof !12

if.then.i.i1092:                                  ; preds = %_ZNSsD2Ev.exit1086
  %_M_refcount.i.i1091 = getelementptr inbounds i8, i8* %268, i64 -8
  %270 = bitcast i8* %_M_refcount.i.i1091 to i32*
  br i1 icmp ne (i8* bitcast (i32 (i32*, void (i8*)*)* @__pthread_key_create to i8*), i8* null), label %if.then.i.i.i1093, label %if.else.i.i.i1095

if.then.i.i.i1093:                                ; preds = %if.then.i.i1092
  %271 = atomicrmw volatile add i32* %270, i32 -1 acq_rel
  br label %invoke.cont.i.i1098

if.else.i.i.i1095:                                ; preds = %if.then.i.i1092
  %272 = load i32, i32* %270, align 4, !tbaa !13
  %add.i.i.i.i1094 = add nsw i32 %272, -1
  store i32 %add.i.i.i.i1094, i32* %270, align 4, !tbaa !13
  br label %invoke.cont.i.i1098

invoke.cont.i.i1098:                              ; preds = %if.else.i.i.i1095, %if.then.i.i.i1093
  %retval.0.i.i.i1096 = phi i32 [ %271, %if.then.i.i.i1093 ], [ %272, %if.else.i.i.i1095 ]
  %cmp3.i.i1097 = icmp slt i32 %retval.0.i.i.i1096, 1
  br i1 %cmp3.i.i1097, label %if.then4.i.i1099, label %_ZNSsD2Ev.exit1100

if.then4.i.i1099:                                 ; preds = %invoke.cont.i.i1098
  call void @_ZNSs4_Rep10_M_destroyERKSaIcE(%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"* nonnull %269, %"class.std::allocator"* nonnull dereferenceable(1) %ref.tmp.i1087)
  br label %_ZNSsD2Ev.exit1100

_ZNSsD2Ev.exit1100:                               ; preds = %if.then4.i.i1099, %invoke.cont.i.i1098, %_ZNSsD2Ev.exit1086
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %53)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %46)
  %273 = load i8*, i8** %_M_p.i.i.i.i1024, align 8, !tbaa !9
  %arrayidx.i.i1103 = getelementptr inbounds i8, i8* %273, i64 -24
  %274 = bitcast i8* %arrayidx.i.i1103 to %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %54)
  %cmp.i.i1104 = icmp eq i8* %arrayidx.i.i1103, bitcast ([0 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE to i8*)
  br i1 %cmp.i.i1104, label %_ZNSsD2Ev.exit1114, label %if.then.i.i1106, !prof !12

if.then.i.i1106:                                  ; preds = %_ZNSsD2Ev.exit1100
  %_M_refcount.i.i1105 = getelementptr inbounds i8, i8* %273, i64 -8
  %275 = bitcast i8* %_M_refcount.i.i1105 to i32*
  br i1 icmp ne (i8* bitcast (i32 (i32*, void (i8*)*)* @__pthread_key_create to i8*), i8* null), label %if.then.i.i.i1107, label %if.else.i.i.i1109

if.then.i.i.i1107:                                ; preds = %if.then.i.i1106
  %276 = atomicrmw volatile add i32* %275, i32 -1 acq_rel
  br label %invoke.cont.i.i1112

if.else.i.i.i1109:                                ; preds = %if.then.i.i1106
  %277 = load i32, i32* %275, align 4, !tbaa !13
  %add.i.i.i.i1108 = add nsw i32 %277, -1
  store i32 %add.i.i.i.i1108, i32* %275, align 4, !tbaa !13
  br label %invoke.cont.i.i1112

invoke.cont.i.i1112:                              ; preds = %if.else.i.i.i1109, %if.then.i.i.i1107
  %retval.0.i.i.i1110 = phi i32 [ %276, %if.then.i.i.i1107 ], [ %277, %if.else.i.i.i1109 ]
  %cmp3.i.i1111 = icmp slt i32 %retval.0.i.i.i1110, 1
  br i1 %cmp3.i.i1111, label %if.then4.i.i1113, label %_ZNSsD2Ev.exit1114

if.then4.i.i1113:                                 ; preds = %invoke.cont.i.i1112
  call void @_ZNSs4_Rep10_M_destroyERKSaIcE(%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"* nonnull %274, %"class.std::allocator"* nonnull dereferenceable(1) %ref.tmp.i1101)
  br label %_ZNSsD2Ev.exit1114

_ZNSsD2Ev.exit1114:                               ; preds = %if.then4.i.i1113, %invoke.cont.i.i1112, %_ZNSsD2Ev.exit1100
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %54)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %45)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %44)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %43)
  %278 = load %"struct.cv::UMatData"*, %"struct.cv::UMatData"** %u.i, align 8, !tbaa !76
  %tobool.i1115 = icmp eq %"struct.cv::UMatData"* %278, null
  br i1 %tobool.i1115, label %if.end.i1119, label %if.then2.i

if.then2.i:                                       ; preds = %_ZNSsD2Ev.exit1114
  %refcount.i = getelementptr inbounds %"struct.cv::UMatData", %"struct.cv::UMatData"* %278, i64 0, i32 3
  %279 = atomicrmw add i32* %refcount.i, i32 1 acq_rel
  br label %if.end.i1119

if.end.i1119:                                     ; preds = %if.then2.i, %_ZNSsD2Ev.exit1114
  %280 = load %"struct.cv::UMatData"*, %"struct.cv::UMatData"** %u.i.i1117, align 8, !tbaa !76
  %tobool.i.i1118 = icmp eq %"struct.cv::UMatData"* %280, null
  br i1 %tobool.i.i1118, label %if.end.i.i1127, label %land.lhs.true.i.i1122

land.lhs.true.i.i1122:                            ; preds = %if.end.i1119
  %refcount.i.i1120 = getelementptr inbounds %"struct.cv::UMatData", %"struct.cv::UMatData"* %280, i64 0, i32 3
  %281 = atomicrmw add i32* %refcount.i.i1120, i32 -1 acq_rel
  %cmp.i.i1121 = icmp eq i32 %281, 1
  br i1 %cmp.i.i1121, label %if.then.i.i1123, label %if.end.i.i1127

if.then.i.i1123:                                  ; preds = %land.lhs.true.i.i1122
  invoke void @_ZN2cv3Mat10deallocateEv(%"class.cv::Mat"* nonnull %last_img)
          to label %if.end.i.i1127 unwind label %lpad25

if.end.i.i1127:                                   ; preds = %if.then.i.i1123, %land.lhs.true.i.i1122, %if.end.i1119
  store %"struct.cv::UMatData"* null, %"struct.cv::UMatData"** %u.i.i1117, align 8, !tbaa !76
  call void @llvm.memset.p0i8.i64(i8* nonnull %55, i8 0, i64 32, i32 8, i1 false)
  %282 = load i32, i32* %dims.i, align 4, !tbaa !24
  %cmp48.i.i1126 = icmp sgt i32 %282, 0
  br i1 %cmp48.i.i1126, label %for.body.lr.ph.i.i1129, label %_ZN2cv3Mat7releaseEv.exit.thread.i

_ZN2cv3Mat7releaseEv.exit.thread.i:               ; preds = %if.end.i.i1127
  %283 = load i32, i32* %flags50.i, align 8, !tbaa !16
  store i32 %283, i32* %flags.i, align 8, !tbaa !16
  br label %land.lhs.true.i1136

for.body.lr.ph.i.i1129:                           ; preds = %if.end.i.i1127
  %284 = load i32*, i32** %p.i.i603, align 8, !tbaa !77
  br label %for.body.i.i1134

for.body.i.i1134:                                 ; preds = %for.body.i.i1134, %for.body.lr.ph.i.i1129
  %indvars.iv.i.i1130 = phi i64 [ 0, %for.body.lr.ph.i.i1129 ], [ %indvars.iv.next.i.i1132, %for.body.i.i1134 ]
  %arrayidx.i.i1131 = getelementptr inbounds i32, i32* %284, i64 %indvars.iv.i.i1130
  store i32 0, i32* %arrayidx.i.i1131, align 4, !tbaa !13
  %indvars.iv.next.i.i1132 = add nuw nsw i64 %indvars.iv.i.i1130, 1
  %285 = load i32, i32* %dims.i, align 4, !tbaa !24
  %286 = sext i32 %285 to i64
  %cmp4.i.i1133 = icmp slt i64 %indvars.iv.next.i.i1132, %286
  br i1 %cmp4.i.i1133, label %for.body.i.i1134, label %_ZN2cv3Mat7releaseEv.exit.i

_ZN2cv3Mat7releaseEv.exit.i:                      ; preds = %for.body.i.i1134
  %287 = load i32, i32* %flags50.i, align 8, !tbaa !16
  store i32 %287, i32* %flags.i, align 8, !tbaa !16
  %cmp5.i = icmp slt i32 %285, 3
  br i1 %cmp5.i, label %land.lhs.true.i1136, label %if.else.i

land.lhs.true.i1136:                              ; preds = %_ZN2cv3Mat7releaseEv.exit.i, %_ZN2cv3Mat7releaseEv.exit.thread.i
  %288 = load i32, i32* %dims6.i, align 4, !tbaa !24
  %cmp7.i = icmp slt i32 %288, 3
  br i1 %cmp7.i, label %if.then8.i, label %if.else.i

if.then8.i:                                       ; preds = %land.lhs.true.i1136
  store i32 %288, i32* %dims.i, align 4, !tbaa !24
  %289 = load i32, i32* %rows, align 8, !tbaa !22
  store i32 %289, i32* %rows.i, align 8, !tbaa !22
  %290 = load i32, i32* %cols, align 4, !tbaa !23
  store i32 %290, i32* %cols12.i, align 4, !tbaa !23
  %291 = load i64*, i64** %p.i736, align 8, !tbaa !21
  %292 = load i64, i64* %291, align 8, !tbaa !25
  %293 = load i64*, i64** %p.i3.i, align 8, !tbaa !21
  store i64 %292, i64* %293, align 8, !tbaa !25
  %arrayidx.i48.i = getelementptr inbounds i64, i64* %291, i64 1
  %294 = load i64, i64* %arrayidx.i48.i, align 8, !tbaa !25
  %arrayidx.i46.i = getelementptr inbounds i64, i64* %293, i64 1
  store i64 %294, i64* %arrayidx.i46.i, align 8, !tbaa !25
  br label %invoke.cont335

if.else.i:                                        ; preds = %land.lhs.true.i1136, %_ZN2cv3Mat7releaseEv.exit.i
  invoke void @_ZN2cv3Mat8copySizeERKS0_(%"class.cv::Mat"* nonnull %last_img, %"class.cv::Mat"* nonnull dereferenceable(96) %img)
          to label %invoke.cont335 unwind label %lpad25

invoke.cont335:                                   ; preds = %if.else.i, %if.then8.i
  %295 = load <4 x i64>, <4 x i64>* %71, align 8, !tbaa !15
  store <4 x i64> %295, <4 x i64>* %72, align 8, !tbaa !15
  %296 = load i64, i64* %56, align 8, !tbaa !78
  store i64 %296, i64* %57, align 8, !tbaa !78
  %297 = load i64, i64* %58, align 8, !tbaa !76
  store i64 %297, i64* %59, align 8, !tbaa !76
  %298 = load %"struct.cv::UMatData"*, %"struct.cv::UMatData"** %u.i1142, align 8, !tbaa !76
  %tobool.i1143 = icmp eq %"struct.cv::UMatData"* %298, null
  %299 = inttoptr i64 %297 to %"struct.cv::UMatData"*
  br i1 %tobool.i1143, label %if.end.i1149, label %if.then2.i1146

if.then2.i1146:                                   ; preds = %invoke.cont335
  %refcount.i1145 = getelementptr inbounds %"struct.cv::UMatData", %"struct.cv::UMatData"* %298, i64 0, i32 3
  %300 = atomicrmw add i32* %refcount.i1145, i32 1 acq_rel
  %.pre1887 = load %"struct.cv::UMatData"*, %"struct.cv::UMatData"** %u.i, align 8, !tbaa !76
  br label %if.end.i1149

if.end.i1149:                                     ; preds = %if.then2.i1146, %invoke.cont335
  %301 = phi %"struct.cv::UMatData"* [ %.pre1887, %if.then2.i1146 ], [ %299, %invoke.cont335 ]
  %tobool.i.i1148 = icmp eq %"struct.cv::UMatData"* %301, null
  br i1 %tobool.i.i1148, label %if.end.i.i1157, label %land.lhs.true.i.i1152

land.lhs.true.i.i1152:                            ; preds = %if.end.i1149
  %refcount.i.i1150 = getelementptr inbounds %"struct.cv::UMatData", %"struct.cv::UMatData"* %301, i64 0, i32 3
  %302 = atomicrmw add i32* %refcount.i.i1150, i32 -1 acq_rel
  %cmp.i.i1151 = icmp eq i32 %302, 1
  br i1 %cmp.i.i1151, label %if.then.i.i1153, label %if.end.i.i1157

if.then.i.i1153:                                  ; preds = %land.lhs.true.i.i1152
  invoke void @_ZN2cv3Mat10deallocateEv(%"class.cv::Mat"* nonnull %img)
          to label %if.end.i.i1157 unwind label %lpad25

if.end.i.i1157:                                   ; preds = %if.then.i.i1153, %land.lhs.true.i.i1152, %if.end.i1149
  store %"struct.cv::UMatData"* null, %"struct.cv::UMatData"** %u.i, align 8, !tbaa !76
  call void @llvm.memset.p0i8.i64(i8* nonnull %60, i8 0, i64 32, i32 8, i1 false)
  %303 = load i32, i32* %dims6.i, align 4, !tbaa !24
  %cmp48.i.i1156 = icmp sgt i32 %303, 0
  br i1 %cmp48.i.i1156, label %for.body.lr.ph.i.i1162, label %_ZN2cv3Mat7releaseEv.exit.thread.i1160

_ZN2cv3Mat7releaseEv.exit.thread.i1160:           ; preds = %if.end.i.i1157
  %304 = load i32, i32* %flags50.i1158, align 16, !tbaa !16
  store i32 %304, i32* %flags50.i, align 8, !tbaa !16
  br label %land.lhs.true.i1174

for.body.lr.ph.i.i1162:                           ; preds = %if.end.i.i1157
  %305 = load i32*, i32** %p.i.i1161, align 8, !tbaa !77
  br label %for.body.i.i1167

for.body.i.i1167:                                 ; preds = %for.body.i.i1167, %for.body.lr.ph.i.i1162
  %indvars.iv.i.i1163 = phi i64 [ 0, %for.body.lr.ph.i.i1162 ], [ %indvars.iv.next.i.i1165, %for.body.i.i1167 ]
  %arrayidx.i.i1164 = getelementptr inbounds i32, i32* %305, i64 %indvars.iv.i.i1163
  store i32 0, i32* %arrayidx.i.i1164, align 4, !tbaa !13
  %indvars.iv.next.i.i1165 = add nuw nsw i64 %indvars.iv.i.i1163, 1
  %306 = load i32, i32* %dims6.i, align 4, !tbaa !24
  %307 = sext i32 %306 to i64
  %cmp4.i.i1166 = icmp slt i64 %indvars.iv.next.i.i1165, %307
  br i1 %cmp4.i.i1166, label %for.body.i.i1167, label %_ZN2cv3Mat7releaseEv.exit.i1171

_ZN2cv3Mat7releaseEv.exit.i1171:                  ; preds = %for.body.i.i1167
  %308 = load i32, i32* %flags50.i1158, align 16, !tbaa !16
  store i32 %308, i32* %flags50.i, align 8, !tbaa !16
  %cmp5.i1170 = icmp slt i32 %306, 3
  br i1 %cmp5.i1170, label %land.lhs.true.i1174, label %if.else.i1184

land.lhs.true.i1174:                              ; preds = %_ZN2cv3Mat7releaseEv.exit.i1171, %_ZN2cv3Mat7releaseEv.exit.thread.i1160
  %309 = load i32, i32* %dims6.i1172, align 4, !tbaa !24
  %cmp7.i1173 = icmp slt i32 %309, 3
  br i1 %cmp7.i1173, label %if.then8.i1183, label %if.else.i1184

if.then8.i1183:                                   ; preds = %land.lhs.true.i1174
  store i32 %309, i32* %dims6.i, align 4, !tbaa !24
  %310 = load i32, i32* %rows.i1175, align 8, !tbaa !22
  store i32 %310, i32* %rows, align 8, !tbaa !22
  %311 = load i32, i32* %cols.i1177, align 4, !tbaa !23
  store i32 %311, i32* %cols, align 4, !tbaa !23
  %312 = load i64*, i64** %p.i757, align 8, !tbaa !21
  %313 = load i64, i64* %312, align 8, !tbaa !25
  %314 = load i64*, i64** %p.i736, align 8, !tbaa !21
  store i64 %313, i64* %314, align 8, !tbaa !25
  %arrayidx.i48.i1181 = getelementptr inbounds i64, i64* %312, i64 1
  %315 = load i64, i64* %arrayidx.i48.i1181, align 8, !tbaa !25
  %arrayidx.i46.i1182 = getelementptr inbounds i64, i64* %314, i64 1
  store i64 %315, i64* %arrayidx.i46.i1182, align 8, !tbaa !25
  br label %invoke.cont337

if.else.i1184:                                    ; preds = %land.lhs.true.i1174, %_ZN2cv3Mat7releaseEv.exit.i1171
  invoke void @_ZN2cv3Mat8copySizeERKS0_(%"class.cv::Mat"* nonnull %img, %"class.cv::Mat"* nonnull dereferenceable(96) %next_section_img)
          to label %invoke.cont337 unwind label %lpad25

invoke.cont337:                                   ; preds = %if.else.i1184, %if.then8.i1183
  %316 = load <4 x i64>, <4 x i64>* %73, align 16, !tbaa !15
  store <4 x i64> %316, <4 x i64>* %74, align 8, !tbaa !15
  %317 = load i64, i64* %61, align 16, !tbaa !78
  store i64 %317, i64* %56, align 8, !tbaa !78
  %318 = load i64, i64* %62, align 8, !tbaa !76
  store i64 %318, i64* %58, align 8, !tbaa !76
  %indvars.iv.next1885 = add nuw i64 %indvars.iv1884, 1
  %319 = load i64, i64* %0, align 8, !tbaa !0
  %320 = load i64, i64* %2, align 8, !tbaa !6
  %sub.ptr.sub.i1198 = sub i64 %319, %320
  %sub.ptr.div.i1199 = ashr exact i64 %sub.ptr.sub.i1198, 3
  %cmp343 = icmp ugt i64 %sub.ptr.div.i1199, %indvars.iv.next1885
  br i1 %cmp343, label %invoke.cont352, label %if.end359

invoke.cont352:                                   ; preds = %invoke.cont337
  %321 = inttoptr i64 %320 to %"class.tfk::Section"**
  call void @llvm.lifetime.start.p0i8(i64 96, i8* nonnull %63)
  %add.ptr.i1201 = getelementptr inbounds %"class.tfk::Section"*, %"class.tfk::Section"** %321, i64 %indvars.iv.next1885
  %322 = load %"class.tfk::Section"*, %"class.tfk::Section"** %add.ptr.i1201, align 8, !tbaa !15
  %323 = load <4 x i32>, <4 x i32>* %75, align 4, !tbaa !7
  store <4 x i32> %323, <4 x i32>* %agg.tmp350, align 16, !tbaa !7
  invoke void @_ZN3tfk6Render6renderEPNS_7SectionESt4pairIN2cv6Point_IfEES6_ENS_10ResolutionEb(%"class.cv::Mat"* nonnull sret %ref.tmp345, %"class.tfk::Render"* %this, %"class.tfk::Section"* %322, %"struct.std::pair"* nonnull %tmpcast1941, i32 %resolution, i1 zeroext false)
          to label %invoke.cont353 unwind label %lpad351

invoke.cont353:                                   ; preds = %invoke.cont352
  %324 = load %"struct.cv::UMatData"*, %"struct.cv::UMatData"** %u.i1142, align 8, !tbaa !76
  %tobool.i.i1209 = icmp eq %"struct.cv::UMatData"* %324, null
  br i1 %tobool.i.i1209, label %if.end.i.i1218, label %land.lhs.true.i.i1213

land.lhs.true.i.i1213:                            ; preds = %invoke.cont353
  %refcount.i.i1211 = getelementptr inbounds %"struct.cv::UMatData", %"struct.cv::UMatData"* %324, i64 0, i32 3
  %325 = atomicrmw add i32* %refcount.i.i1211, i32 -1 acq_rel
  %cmp.i.i1212 = icmp eq i32 %325, 1
  br i1 %cmp.i.i1212, label %if.then.i.i1214, label %if.end.i.i1218

if.then.i.i1214:                                  ; preds = %land.lhs.true.i.i1213
  invoke void @_ZN2cv3Mat10deallocateEv(%"class.cv::Mat"* nonnull %next_section_img)
          to label %if.end.i.i1218 unwind label %lpad354

if.end.i.i1218:                                   ; preds = %if.then.i.i1214, %land.lhs.true.i.i1213, %invoke.cont353
  store %"struct.cv::UMatData"* null, %"struct.cv::UMatData"** %u.i1142, align 8, !tbaa !76
  call void @llvm.memset.p0i8.i64(i8* nonnull %65, i8 0, i64 32, i32 16, i1 false)
  %326 = load i32, i32* %dims6.i1172, align 4, !tbaa !24
  %cmp48.i.i1217 = icmp sgt i32 %326, 0
  br i1 %cmp48.i.i1217, label %for.body.lr.ph.i.i1220, label %_ZN2cv3Mat7releaseEv.exit.i1238

for.body.lr.ph.i.i1220:                           ; preds = %if.end.i.i1218
  %327 = load i32*, i32** %p.i.i1219, align 16, !tbaa !77
  br label %for.body.i.i1225

for.body.i.i1225:                                 ; preds = %for.body.i.i1225, %for.body.lr.ph.i.i1220
  %indvars.iv.i.i1221 = phi i64 [ 0, %for.body.lr.ph.i.i1220 ], [ %indvars.iv.next.i.i1223, %for.body.i.i1225 ]
  %arrayidx.i.i1222 = getelementptr inbounds i32, i32* %327, i64 %indvars.iv.i.i1221
  store i32 0, i32* %arrayidx.i.i1222, align 4, !tbaa !13
  %indvars.iv.next.i.i1223 = add nuw nsw i64 %indvars.iv.i.i1221, 1
  %328 = load i32, i32* %dims6.i1172, align 4, !tbaa !24
  %329 = sext i32 %328 to i64
  %cmp4.i.i1224 = icmp slt i64 %indvars.iv.next.i.i1223, %329
  br i1 %cmp4.i.i1224, label %for.body.i.i1225, label %_ZN2cv3Mat7releaseEv.exit.i1238

_ZN2cv3Mat7releaseEv.exit.i1238:                  ; preds = %for.body.i.i1225, %if.end.i.i1218
  %330 = load <4 x i32>, <4 x i32>* %76, align 16, !tbaa !13
  store <4 x i32> %330, <4 x i32>* %77, align 16, !tbaa !13
  %331 = load <4 x i64>, <4 x i64>* %78, align 16, !tbaa !15
  store <4 x i64> %331, <4 x i64>* %79, align 16, !tbaa !15
  %332 = load i64, i64* %66, align 16, !tbaa !78
  store i64 %332, i64* %61, align 16, !tbaa !78
  %333 = load i64, i64* %67, align 8, !tbaa !76
  store i64 %333, i64* %62, align 8, !tbaa !76
  %334 = load i64*, i64** %p.i1236, align 8, !tbaa !34
  %cmp13.i = icmp eq i64* %334, %arraydecay.i1237
  %335 = extractelement <4 x i32> %330, i32 1
  br i1 %cmp13.i, label %if.end24.i, label %if.then14.i

if.then14.i:                                      ; preds = %_ZN2cv3Mat7releaseEv.exit.i1238
  %336 = bitcast i64* %334 to i8*
  invoke void @_ZN2cv8fastFreeEPv(i8* %336)
          to label %.noexc1241 unwind label %lpad354

.noexc1241:                                       ; preds = %if.then14.i
  store i64* %arraydecay.i1237, i64** %p.i1236, align 8, !tbaa !34
  store i32* %rows.i1175, i32** %p.i.i1219, align 16, !tbaa !77
  %.pre1888 = load i32, i32* %dims.i1227, align 4, !tbaa !24
  br label %if.end24.i

if.end24.i:                                       ; preds = %.noexc1241, %_ZN2cv3Mat7releaseEv.exit.i1238
  %337 = phi i32 [ %.pre1888, %.noexc1241 ], [ %335, %_ZN2cv3Mat7releaseEv.exit.i1238 ]
  %cmp26.i = icmp slt i32 %337, 3
  br i1 %cmp26.i, label %if.then27.i, label %if.else.i1239

if.then27.i:                                      ; preds = %if.end24.i
  %338 = load i64*, i64** %p.i90.i, align 8, !tbaa !21
  %339 = load i64, i64* %338, align 8, !tbaa !25
  store i64 %339, i64* %arraydecay.i1237, align 8, !tbaa !25
  %arrayidx.i94.i = getelementptr inbounds i64, i64* %338, i64 1
  %340 = load i64, i64* %arrayidx.i94.i, align 8, !tbaa !25
  store i64 %340, i64* %arrayidx.i92.i, align 8, !tbaa !25
  br label %invoke.cont.i1262

if.else.i1239:                                    ; preds = %if.end24.i
  %341 = load <2 x i64>, <2 x i64>* %80, align 16, !tbaa !15
  store <2 x i64> %341, <2 x i64>* %81, align 16, !tbaa !15
  store i64* %arraydecay45.i, i64** %p.i90.i, align 8, !tbaa !34
  store i32* %rows.i1228, i32** %p40.i, align 16, !tbaa !77
  br label %invoke.cont.i1262

invoke.cont.i1262:                                ; preds = %if.else.i1239, %if.then27.i
  %342 = phi i64* [ %arraydecay45.i, %if.else.i1239 ], [ %338, %if.then27.i ]
  store i32 1124007936, i32* %flags.i1226, align 16, !tbaa !16
  call void @llvm.memset.p0i8.i64(i8* nonnull %69, i8 0, i64 52, i32 4, i1 false)
  store %"struct.cv::UMatData"* null, %"struct.cv::UMatData"** %u.i1235, align 8, !tbaa !76
  call void @llvm.memset.p0i8.i64(i8* nonnull %70, i8 0, i64 32, i32 16, i1 false)
  %cmp.i1261 = icmp eq i64* %342, %arraydecay45.i
  br i1 %cmp.i1261, label %_ZN2cv3MatD2Ev.exit1266, label %if.then.i1263

if.then.i1263:                                    ; preds = %invoke.cont.i1262
  %343 = bitcast i64* %342 to i8*
  invoke void @_ZN2cv8fastFreeEPv(i8* %343)
          to label %_ZN2cv3MatD2Ev.exit1266 unwind label %terminate.lpad.i1265

terminate.lpad.i1265:                             ; preds = %if.then.i1263
  %344 = landingpad { i8*, i32 }
          catch i8* null
  %345 = extractvalue { i8*, i32 } %344, 0
  call void @__clang_call_terminate(i8* %345)
  unreachable

_ZN2cv3MatD2Ev.exit1266:                          ; preds = %if.then.i1263, %invoke.cont.i1262
  call void @llvm.lifetime.end.p0i8(i64 96, i8* nonnull %63)
  br label %if.end359

lpad303:                                          ; preds = %sync.continue293
  %346 = landingpad { i8*, i32 }
          cleanup
  br label %lpad303.body

lpad303.body:                                     ; preds = %lpad303, %_ZNSsD2Ev.exit.i1018
  %eh.lpad-body1020 = phi { i8*, i32 } [ %346, %lpad303 ], [ %234, %_ZNSsD2Ev.exit.i1018 ]
  %347 = extractvalue { i8*, i32 } %eh.lpad-body1020, 0
  %348 = extractvalue { i8*, i32 } %eh.lpad-body1020, 1
  br label %ehcleanup331

lpad306:                                          ; preds = %invoke.cont304
  %349 = landingpad { i8*, i32 }
          cleanup
  %350 = extractvalue { i8*, i32 } %349, 0
  %351 = extractvalue { i8*, i32 } %349, 1
  br label %ehcleanup329

lpad308:                                          ; preds = %cond.false.i1037, %cond.true.i1036
  %352 = landingpad { i8*, i32 }
          cleanup
  %353 = extractvalue { i8*, i32 } %352, 0
  %354 = extractvalue { i8*, i32 } %352, 1
  br label %ehcleanup328

lpad310:                                          ; preds = %invoke.cont309
  %355 = landingpad { i8*, i32 }
          cleanup
  %356 = extractvalue { i8*, i32 } %355, 0
  %357 = extractvalue { i8*, i32 } %355, 1
  br label %ehcleanup327

lpad312:                                          ; preds = %if.then.i1050
  %358 = landingpad { i8*, i32 }
          cleanup
  %359 = extractvalue { i8*, i32 } %358, 0
  %360 = extractvalue { i8*, i32 } %358, 1
  br label %ehcleanup326

lpad318:                                          ; preds = %invoke.cont316
  %361 = landingpad { i8*, i32 }
          cleanup
  %362 = extractvalue { i8*, i32 } %361, 0
  %363 = extractvalue { i8*, i32 } %361, 1
  %364 = load i32*, i32** %_M_start.i.i, align 8, !tbaa !73
  %tobool.i.i.i1268 = icmp eq i32* %364, null
  br i1 %tobool.i.i.i1268, label %ehcleanup324, label %if.then.i.i.i1270

if.then.i.i.i1270:                                ; preds = %lpad318
  %365 = bitcast i32* %364 to i8*
  call void @_ZdlPv(i8* %365)
  br label %ehcleanup324

ehcleanup324:                                     ; preds = %if.then.i.i.i1270, %lpad318
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %50)
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %48)
  invoke void @_ZN2cv6String10deallocateEv(%"class.cv::String"* nonnull %ref.tmp)
          to label %ehcleanup326 unwind label %terminate.lpad.i1273

terminate.lpad.i1273:                             ; preds = %ehcleanup324
  %366 = landingpad { i8*, i32 }
          catch i8* null
  %367 = extractvalue { i8*, i32 } %366, 0
  call void @__clang_call_terminate(i8* %367)
  unreachable

ehcleanup326:                                     ; preds = %ehcleanup324, %lpad312
  %ehselector.slot.1 = phi i32 [ %360, %lpad312 ], [ %363, %ehcleanup324 ]
  %exn.slot.1 = phi i8* [ %359, %lpad312 ], [ %362, %ehcleanup324 ]
  %368 = bitcast i8* %arrayidx.i.i.i.i to %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"*
  %369 = getelementptr inbounds %"class.std::allocator", %"class.std::allocator"* %ref.tmp.i1275, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %369)
  %cmp.i.i1278 = icmp eq i8* %arrayidx.i.i.i.i, bitcast ([0 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE to i8*)
  br i1 %cmp.i.i1278, label %_ZNSsD2Ev.exit1288, label %if.then.i.i1280, !prof !12

if.then.i.i1280:                                  ; preds = %ehcleanup326
  %_M_refcount.i.i1279 = getelementptr inbounds i8, i8* %253, i64 -8
  %370 = bitcast i8* %_M_refcount.i.i1279 to i32*
  br i1 icmp ne (i8* bitcast (i32 (i32*, void (i8*)*)* @__pthread_key_create to i8*), i8* null), label %if.then.i.i.i1281, label %if.else.i.i.i1283

if.then.i.i.i1281:                                ; preds = %if.then.i.i1280
  %371 = atomicrmw volatile add i32* %370, i32 -1 acq_rel
  br label %invoke.cont.i.i1286

if.else.i.i.i1283:                                ; preds = %if.then.i.i1280
  %372 = load i32, i32* %370, align 4, !tbaa !13
  %add.i.i.i.i1282 = add nsw i32 %372, -1
  store i32 %add.i.i.i.i1282, i32* %370, align 4, !tbaa !13
  br label %invoke.cont.i.i1286

invoke.cont.i.i1286:                              ; preds = %if.else.i.i.i1283, %if.then.i.i.i1281
  %retval.0.i.i.i1284 = phi i32 [ %371, %if.then.i.i.i1281 ], [ %372, %if.else.i.i.i1283 ]
  %cmp3.i.i1285 = icmp slt i32 %retval.0.i.i.i1284, 1
  br i1 %cmp3.i.i1285, label %if.then4.i.i1287, label %_ZNSsD2Ev.exit1288

if.then4.i.i1287:                                 ; preds = %invoke.cont.i.i1286
  call void @_ZNSs4_Rep10_M_destroyERKSaIcE(%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"* nonnull %368, %"class.std::allocator"* nonnull dereferenceable(1) %ref.tmp.i1275)
  br label %_ZNSsD2Ev.exit1288

_ZNSsD2Ev.exit1288:                               ; preds = %if.then4.i.i1287, %invoke.cont.i.i1286, %ehcleanup326
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %369)
  br label %ehcleanup327

ehcleanup327:                                     ; preds = %_ZNSsD2Ev.exit1288, %lpad310
  %ehselector.slot.2 = phi i32 [ %ehselector.slot.1, %_ZNSsD2Ev.exit1288 ], [ %357, %lpad310 ]
  %exn.slot.2 = phi i8* [ %exn.slot.1, %_ZNSsD2Ev.exit1288 ], [ %356, %lpad310 ]
  %373 = load i8*, i8** %_M_p.i.i.i1074, align 8, !tbaa !9
  %arrayidx.i.i1291 = getelementptr inbounds i8, i8* %373, i64 -24
  %374 = bitcast i8* %arrayidx.i.i1291 to %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"*
  %375 = getelementptr inbounds %"class.std::allocator", %"class.std::allocator"* %ref.tmp.i1289, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %375)
  %cmp.i.i1292 = icmp eq i8* %arrayidx.i.i1291, bitcast ([0 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE to i8*)
  br i1 %cmp.i.i1292, label %_ZNSsD2Ev.exit1302, label %if.then.i.i1294, !prof !12

if.then.i.i1294:                                  ; preds = %ehcleanup327
  %_M_refcount.i.i1293 = getelementptr inbounds i8, i8* %373, i64 -8
  %376 = bitcast i8* %_M_refcount.i.i1293 to i32*
  br i1 icmp ne (i8* bitcast (i32 (i32*, void (i8*)*)* @__pthread_key_create to i8*), i8* null), label %if.then.i.i.i1295, label %if.else.i.i.i1297

if.then.i.i.i1295:                                ; preds = %if.then.i.i1294
  %377 = atomicrmw volatile add i32* %376, i32 -1 acq_rel
  br label %invoke.cont.i.i1300

if.else.i.i.i1297:                                ; preds = %if.then.i.i1294
  %378 = load i32, i32* %376, align 4, !tbaa !13
  %add.i.i.i.i1296 = add nsw i32 %378, -1
  store i32 %add.i.i.i.i1296, i32* %376, align 4, !tbaa !13
  br label %invoke.cont.i.i1300

invoke.cont.i.i1300:                              ; preds = %if.else.i.i.i1297, %if.then.i.i.i1295
  %retval.0.i.i.i1298 = phi i32 [ %377, %if.then.i.i.i1295 ], [ %378, %if.else.i.i.i1297 ]
  %cmp3.i.i1299 = icmp slt i32 %retval.0.i.i.i1298, 1
  br i1 %cmp3.i.i1299, label %if.then4.i.i1301, label %_ZNSsD2Ev.exit1302

if.then4.i.i1301:                                 ; preds = %invoke.cont.i.i1300
  call void @_ZNSs4_Rep10_M_destroyERKSaIcE(%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"* nonnull %374, %"class.std::allocator"* nonnull dereferenceable(1) %ref.tmp.i1289)
  br label %_ZNSsD2Ev.exit1302

_ZNSsD2Ev.exit1302:                               ; preds = %if.then4.i.i1301, %invoke.cont.i.i1300, %ehcleanup327
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %375)
  br label %ehcleanup328

ehcleanup328:                                     ; preds = %_ZNSsD2Ev.exit1302, %lpad308
  %ehselector.slot.3 = phi i32 [ %ehselector.slot.2, %_ZNSsD2Ev.exit1302 ], [ %354, %lpad308 ]
  %exn.slot.3 = phi i8* [ %exn.slot.2, %_ZNSsD2Ev.exit1302 ], [ %353, %lpad308 ]
  %379 = load i8*, i8** %_M_p.i.i.i17.i1027, align 8, !tbaa !9
  %arrayidx.i.i1305 = getelementptr inbounds i8, i8* %379, i64 -24
  %380 = bitcast i8* %arrayidx.i.i1305 to %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"*
  %381 = getelementptr inbounds %"class.std::allocator", %"class.std::allocator"* %ref.tmp.i1303, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %381)
  %cmp.i.i1306 = icmp eq i8* %arrayidx.i.i1305, bitcast ([0 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE to i8*)
  br i1 %cmp.i.i1306, label %_ZNSsD2Ev.exit1316, label %if.then.i.i1308, !prof !12

if.then.i.i1308:                                  ; preds = %ehcleanup328
  %_M_refcount.i.i1307 = getelementptr inbounds i8, i8* %379, i64 -8
  %382 = bitcast i8* %_M_refcount.i.i1307 to i32*
  br i1 icmp ne (i8* bitcast (i32 (i32*, void (i8*)*)* @__pthread_key_create to i8*), i8* null), label %if.then.i.i.i1309, label %if.else.i.i.i1311

if.then.i.i.i1309:                                ; preds = %if.then.i.i1308
  %383 = atomicrmw volatile add i32* %382, i32 -1 acq_rel
  br label %invoke.cont.i.i1314

if.else.i.i.i1311:                                ; preds = %if.then.i.i1308
  %384 = load i32, i32* %382, align 4, !tbaa !13
  %add.i.i.i.i1310 = add nsw i32 %384, -1
  store i32 %add.i.i.i.i1310, i32* %382, align 4, !tbaa !13
  br label %invoke.cont.i.i1314

invoke.cont.i.i1314:                              ; preds = %if.else.i.i.i1311, %if.then.i.i.i1309
  %retval.0.i.i.i1312 = phi i32 [ %383, %if.then.i.i.i1309 ], [ %384, %if.else.i.i.i1311 ]
  %cmp3.i.i1313 = icmp slt i32 %retval.0.i.i.i1312, 1
  br i1 %cmp3.i.i1313, label %if.then4.i.i1315, label %_ZNSsD2Ev.exit1316

if.then4.i.i1315:                                 ; preds = %invoke.cont.i.i1314
  call void @_ZNSs4_Rep10_M_destroyERKSaIcE(%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"* nonnull %380, %"class.std::allocator"* nonnull dereferenceable(1) %ref.tmp.i1303)
  br label %_ZNSsD2Ev.exit1316

_ZNSsD2Ev.exit1316:                               ; preds = %if.then4.i.i1315, %invoke.cont.i.i1314, %ehcleanup328
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %381)
  br label %ehcleanup329

ehcleanup329:                                     ; preds = %_ZNSsD2Ev.exit1316, %lpad306
  %ehselector.slot.4 = phi i32 [ %ehselector.slot.3, %_ZNSsD2Ev.exit1316 ], [ %351, %lpad306 ]
  %exn.slot.4 = phi i8* [ %exn.slot.3, %_ZNSsD2Ev.exit1316 ], [ %350, %lpad306 ]
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %46)
  %385 = load i8*, i8** %_M_p.i.i.i.i1024, align 8, !tbaa !9
  %arrayidx.i.i1319 = getelementptr inbounds i8, i8* %385, i64 -24
  %386 = bitcast i8* %arrayidx.i.i1319 to %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"*
  %387 = getelementptr inbounds %"class.std::allocator", %"class.std::allocator"* %ref.tmp.i1317, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %387)
  %cmp.i.i1320 = icmp eq i8* %arrayidx.i.i1319, bitcast ([0 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE to i8*)
  br i1 %cmp.i.i1320, label %_ZNSsD2Ev.exit1330, label %if.then.i.i1322, !prof !12

if.then.i.i1322:                                  ; preds = %ehcleanup329
  %_M_refcount.i.i1321 = getelementptr inbounds i8, i8* %385, i64 -8
  %388 = bitcast i8* %_M_refcount.i.i1321 to i32*
  br i1 icmp ne (i8* bitcast (i32 (i32*, void (i8*)*)* @__pthread_key_create to i8*), i8* null), label %if.then.i.i.i1323, label %if.else.i.i.i1325

if.then.i.i.i1323:                                ; preds = %if.then.i.i1322
  %389 = atomicrmw volatile add i32* %388, i32 -1 acq_rel
  br label %invoke.cont.i.i1328

if.else.i.i.i1325:                                ; preds = %if.then.i.i1322
  %390 = load i32, i32* %388, align 4, !tbaa !13
  %add.i.i.i.i1324 = add nsw i32 %390, -1
  store i32 %add.i.i.i.i1324, i32* %388, align 4, !tbaa !13
  br label %invoke.cont.i.i1328

invoke.cont.i.i1328:                              ; preds = %if.else.i.i.i1325, %if.then.i.i.i1323
  %retval.0.i.i.i1326 = phi i32 [ %389, %if.then.i.i.i1323 ], [ %390, %if.else.i.i.i1325 ]
  %cmp3.i.i1327 = icmp slt i32 %retval.0.i.i.i1326, 1
  br i1 %cmp3.i.i1327, label %if.then4.i.i1329, label %_ZNSsD2Ev.exit1330

if.then4.i.i1329:                                 ; preds = %invoke.cont.i.i1328
  call void @_ZNSs4_Rep10_M_destroyERKSaIcE(%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"* nonnull %386, %"class.std::allocator"* nonnull dereferenceable(1) %ref.tmp.i1317)
  br label %_ZNSsD2Ev.exit1330

_ZNSsD2Ev.exit1330:                               ; preds = %if.then4.i.i1329, %invoke.cont.i.i1328, %ehcleanup329
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %387)
  br label %ehcleanup331

ehcleanup331:                                     ; preds = %_ZNSsD2Ev.exit1330, %lpad303.body
  %ehselector.slot.5 = phi i32 [ %ehselector.slot.4, %_ZNSsD2Ev.exit1330 ], [ %348, %lpad303.body ]
  %exn.slot.5 = phi i8* [ %exn.slot.4, %_ZNSsD2Ev.exit1330 ], [ %347, %lpad303.body ]
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %45)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %44)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %43)
  br label %ehcleanup360

lpad351:                                          ; preds = %invoke.cont352
  %391 = landingpad { i8*, i32 }
          cleanup
  %392 = extractvalue { i8*, i32 } %391, 0
  %393 = extractvalue { i8*, i32 } %391, 1
  br label %ehcleanup358

lpad354:                                          ; preds = %if.then14.i, %if.then.i.i1214
  %394 = landingpad { i8*, i32 }
          cleanup
  %395 = extractvalue { i8*, i32 } %394, 0
  %396 = extractvalue { i8*, i32 } %394, 1
  %397 = load %"struct.cv::UMatData"*, %"struct.cv::UMatData"** %u.i1235, align 8, !tbaa !76
  %tobool.i.i1332 = icmp eq %"struct.cv::UMatData"* %397, null
  br i1 %tobool.i.i1332, label %if.end.i.i1340, label %land.lhs.true.i.i1335

land.lhs.true.i.i1335:                            ; preds = %lpad354
  %refcount.i.i1333 = getelementptr inbounds %"struct.cv::UMatData", %"struct.cv::UMatData"* %397, i64 0, i32 3
  %398 = atomicrmw add i32* %refcount.i.i1333, i32 -1 acq_rel
  %cmp.i.i1334 = icmp eq i32 %398, 1
  br i1 %cmp.i.i1334, label %if.then.i.i1336, label %if.end.i.i1340

if.then.i.i1336:                                  ; preds = %land.lhs.true.i.i1335
  invoke void @_ZN2cv3Mat10deallocateEv(%"class.cv::Mat"* nonnull %ref.tmp345)
          to label %if.end.i.i1340 unwind label %terminate.lpad.i1354

if.end.i.i1340:                                   ; preds = %if.then.i.i1336, %land.lhs.true.i.i1335, %lpad354
  store %"struct.cv::UMatData"* null, %"struct.cv::UMatData"** %u.i1235, align 8, !tbaa !76
  call void @llvm.memset.p0i8.i64(i8* nonnull %70, i8 0, i64 32, i32 16, i1 false)
  %399 = load i32, i32* %dims.i1227, align 4, !tbaa !24
  %cmp48.i.i1339 = icmp sgt i32 %399, 0
  br i1 %cmp48.i.i1339, label %for.body.lr.ph.i.i1342, label %invoke.cont.i1351

for.body.lr.ph.i.i1342:                           ; preds = %if.end.i.i1340
  %400 = load i32*, i32** %p40.i, align 16, !tbaa !77
  br label %for.body.i.i1347

for.body.i.i1347:                                 ; preds = %for.body.i.i1347, %for.body.lr.ph.i.i1342
  %indvars.iv.i.i1343 = phi i64 [ 0, %for.body.lr.ph.i.i1342 ], [ %indvars.iv.next.i.i1345, %for.body.i.i1347 ]
  %arrayidx.i.i1344 = getelementptr inbounds i32, i32* %400, i64 %indvars.iv.i.i1343
  store i32 0, i32* %arrayidx.i.i1344, align 4, !tbaa !13
  %indvars.iv.next.i.i1345 = add nuw nsw i64 %indvars.iv.i.i1343, 1
  %401 = load i32, i32* %dims.i1227, align 4, !tbaa !24
  %402 = sext i32 %401 to i64
  %cmp4.i.i1346 = icmp slt i64 %indvars.iv.next.i.i1345, %402
  br i1 %cmp4.i.i1346, label %for.body.i.i1347, label %invoke.cont.i1351

invoke.cont.i1351:                                ; preds = %for.body.i.i1347, %if.end.i.i1340
  %403 = load i64*, i64** %p.i1259, align 8, !tbaa !34
  %cmp.i1350 = icmp eq i64* %403, %arraydecay45.i
  br i1 %cmp.i1350, label %ehcleanup358, label %if.then.i1352

if.then.i1352:                                    ; preds = %invoke.cont.i1351
  %404 = bitcast i64* %403 to i8*
  invoke void @_ZN2cv8fastFreeEPv(i8* %404)
          to label %ehcleanup358 unwind label %terminate.lpad.i1354

terminate.lpad.i1354:                             ; preds = %if.then.i1352, %if.then.i.i1336
  %405 = landingpad { i8*, i32 }
          catch i8* null
  %406 = extractvalue { i8*, i32 } %405, 0
  call void @__clang_call_terminate(i8* %406)
  unreachable

ehcleanup358:                                     ; preds = %if.then.i1352, %invoke.cont.i1351, %lpad351
  %ehselector.slot.6 = phi i32 [ %393, %lpad351 ], [ %396, %invoke.cont.i1351 ], [ %396, %if.then.i1352 ]
  %exn.slot.6 = phi i8* [ %392, %lpad351 ], [ %395, %invoke.cont.i1351 ], [ %395, %if.then.i1352 ]
  call void @llvm.lifetime.end.p0i8(i64 96, i8* nonnull %63)
  br label %ehcleanup360

if.end359:                                        ; preds = %_ZN2cv3MatD2Ev.exit1266, %invoke.cont337
  %407 = load %"struct.cv::UMatData"*, %"struct.cv::UMatData"** %u.i.i1356, align 8, !tbaa !76
  %tobool.i.i1357 = icmp eq %"struct.cv::UMatData"* %407, null
  br i1 %tobool.i.i1357, label %if.end.i.i1365, label %land.lhs.true.i.i1360

land.lhs.true.i.i1360:                            ; preds = %if.end359
  %refcount.i.i1358 = getelementptr inbounds %"struct.cv::UMatData", %"struct.cv::UMatData"* %407, i64 0, i32 3
  %408 = atomicrmw add i32* %refcount.i.i1358, i32 -1 acq_rel
  %cmp.i.i1359 = icmp eq i32 %408, 1
  br i1 %cmp.i.i1359, label %if.then.i.i1361, label %if.end.i.i1365

if.then.i.i1361:                                  ; preds = %land.lhs.true.i.i1360
  invoke void @_ZN2cv3Mat10deallocateEv(%"class.cv::Mat"* nonnull %section_p_out_count)
          to label %if.end.i.i1365 unwind label %terminate.lpad.i1379

if.end.i.i1365:                                   ; preds = %if.then.i.i1361, %land.lhs.true.i.i1360, %if.end359
  store %"struct.cv::UMatData"* null, %"struct.cv::UMatData"** %u.i.i1356, align 8, !tbaa !76
  call void @llvm.memset.p0i8.i64(i8* nonnull %64, i8 0, i64 32, i32 8, i1 false)
  %409 = load i32, i32* %dims.i654, align 4, !tbaa !24
  %cmp48.i.i1364 = icmp sgt i32 %409, 0
  br i1 %cmp48.i.i1364, label %for.body.lr.ph.i.i1367, label %invoke.cont.i1376

for.body.lr.ph.i.i1367:                           ; preds = %if.end.i.i1365
  %410 = load i32*, i32** %p.i.i656, align 8, !tbaa !77
  br label %for.body.i.i1372

for.body.i.i1372:                                 ; preds = %for.body.i.i1372, %for.body.lr.ph.i.i1367
  %indvars.iv.i.i1368 = phi i64 [ 0, %for.body.lr.ph.i.i1367 ], [ %indvars.iv.next.i.i1370, %for.body.i.i1372 ]
  %arrayidx.i.i1369 = getelementptr inbounds i32, i32* %410, i64 %indvars.iv.i.i1368
  store i32 0, i32* %arrayidx.i.i1369, align 4, !tbaa !13
  %indvars.iv.next.i.i1370 = add nuw nsw i64 %indvars.iv.i.i1368, 1
  %411 = load i32, i32* %dims.i654, align 4, !tbaa !24
  %412 = sext i32 %411 to i64
  %cmp4.i.i1371 = icmp slt i64 %indvars.iv.next.i.i1370, %412
  br i1 %cmp4.i.i1371, label %for.body.i.i1372, label %invoke.cont.i1376

invoke.cont.i1376:                                ; preds = %for.body.i.i1372, %if.end.i.i1365
  %413 = load i64*, i64** %p.i3.i658, align 8, !tbaa !34
  %cmp.i1375 = icmp eq i64* %413, %arraydecay.i.i657
  br i1 %cmp.i1375, label %_ZN2cv3MatD2Ev.exit1380, label %if.then.i1377

if.then.i1377:                                    ; preds = %invoke.cont.i1376
  %414 = bitcast i64* %413 to i8*
  invoke void @_ZN2cv8fastFreeEPv(i8* %414)
          to label %_ZN2cv3MatD2Ev.exit1380 unwind label %terminate.lpad.i1379

terminate.lpad.i1379:                             ; preds = %if.then.i1377, %if.then.i.i1361
  %415 = landingpad { i8*, i32 }
          catch i8* null
  %416 = extractvalue { i8*, i32 } %415, 0
  call void @__clang_call_terminate(i8* %416)
  unreachable

_ZN2cv3MatD2Ev.exit1380:                          ; preds = %if.then.i1377, %invoke.cont.i1376
  call void @llvm.lifetime.end.p0i8(i64 96, i8* nonnull %39)
  %417 = load %"struct.cv::UMatData"*, %"struct.cv::UMatData"** %u.i.i1381, align 8, !tbaa !76
  %tobool.i.i1382 = icmp eq %"struct.cv::UMatData"* %417, null
  br i1 %tobool.i.i1382, label %if.end.i.i1390, label %land.lhs.true.i.i1385

land.lhs.true.i.i1385:                            ; preds = %_ZN2cv3MatD2Ev.exit1380
  %refcount.i.i1383 = getelementptr inbounds %"struct.cv::UMatData", %"struct.cv::UMatData"* %417, i64 0, i32 3
  %418 = atomicrmw add i32* %refcount.i.i1383, i32 -1 acq_rel
  %cmp.i.i1384 = icmp eq i32 %418, 1
  br i1 %cmp.i.i1384, label %if.then.i.i1386, label %if.end.i.i1390

if.then.i.i1386:                                  ; preds = %land.lhs.true.i.i1385
  invoke void @_ZN2cv3Mat10deallocateEv(%"class.cv::Mat"* nonnull %section_p_out_sum)
          to label %if.end.i.i1390 unwind label %terminate.lpad.i1404

if.end.i.i1390:                                   ; preds = %if.then.i.i1386, %land.lhs.true.i.i1385, %_ZN2cv3MatD2Ev.exit1380
  store %"struct.cv::UMatData"* null, %"struct.cv::UMatData"** %u.i.i1381, align 8, !tbaa !76
  call void @llvm.memset.p0i8.i64(i8* nonnull %68, i8 0, i64 32, i32 8, i1 false)
  %419 = load i32, i32* %dims.i634, align 4, !tbaa !24
  %cmp48.i.i1389 = icmp sgt i32 %419, 0
  br i1 %cmp48.i.i1389, label %for.body.lr.ph.i.i1392, label %invoke.cont.i1401

for.body.lr.ph.i.i1392:                           ; preds = %if.end.i.i1390
  %420 = load i32*, i32** %p.i.i636, align 8, !tbaa !77
  br label %for.body.i.i1397

for.body.i.i1397:                                 ; preds = %for.body.i.i1397, %for.body.lr.ph.i.i1392
  %indvars.iv.i.i1393 = phi i64 [ 0, %for.body.lr.ph.i.i1392 ], [ %indvars.iv.next.i.i1395, %for.body.i.i1397 ]
  %arrayidx.i.i1394 = getelementptr inbounds i32, i32* %420, i64 %indvars.iv.i.i1393
  store i32 0, i32* %arrayidx.i.i1394, align 4, !tbaa !13
  %indvars.iv.next.i.i1395 = add nuw nsw i64 %indvars.iv.i.i1393, 1
  %421 = load i32, i32* %dims.i634, align 4, !tbaa !24
  %422 = sext i32 %421 to i64
  %cmp4.i.i1396 = icmp slt i64 %indvars.iv.next.i.i1395, %422
  br i1 %cmp4.i.i1396, label %for.body.i.i1397, label %invoke.cont.i1401

invoke.cont.i1401:                                ; preds = %for.body.i.i1397, %if.end.i.i1390
  %423 = load i64*, i64** %p.i3.i638, align 8, !tbaa !34
  %cmp.i1400 = icmp eq i64* %423, %arraydecay.i.i637
  br i1 %cmp.i1400, label %_ZN2cv3MatD2Ev.exit1405, label %if.then.i1402

if.then.i1402:                                    ; preds = %invoke.cont.i1401
  %424 = bitcast i64* %423 to i8*
  invoke void @_ZN2cv8fastFreeEPv(i8* %424)
          to label %_ZN2cv3MatD2Ev.exit1405 unwind label %terminate.lpad.i1404

terminate.lpad.i1404:                             ; preds = %if.then.i1402, %if.then.i.i1386
  %425 = landingpad { i8*, i32 }
          catch i8* null
  %426 = extractvalue { i8*, i32 } %425, 0
  call void @__clang_call_terminate(i8* %426)
  unreachable

_ZN2cv3MatD2Ev.exit1405:                          ; preds = %if.then.i1402, %invoke.cont.i1401
  call void @llvm.lifetime.end.p0i8(i64 96, i8* nonnull %36)
  %427 = load i64, i64* %0, align 8, !tbaa !0
  %428 = load i64, i64* %2, align 8, !tbaa !6
  %sub.ptr.sub.i629 = sub i64 %427, %428
  %sub.ptr.div.i630 = ashr exact i64 %sub.ptr.sub.i629, 3
  %cmp17 = icmp ugt i64 %sub.ptr.div.i630, %indvars.iv.next1885
  %429 = inttoptr i64 %428 to %"class.tfk::Section"**
  br i1 %cmp17, label %invoke.cont24, label %for.cond.cleanup

ehcleanup360:                                     ; preds = %ehcleanup358, %ehcleanup331, %lpad25
  %ehselector.slot.7 = phi i32 [ %ehselector.slot.6, %ehcleanup358 ], [ %128, %lpad25 ], [ %ehselector.slot.5, %ehcleanup331 ]
  %exn.slot.7 = phi i8* [ %exn.slot.6, %ehcleanup358 ], [ %127, %lpad25 ], [ %exn.slot.5, %ehcleanup331 ]
  %430 = load %"struct.cv::UMatData"*, %"struct.cv::UMatData"** %u.i.i1356, align 8, !tbaa !76
  %tobool.i.i1407 = icmp eq %"struct.cv::UMatData"* %430, null
  br i1 %tobool.i.i1407, label %if.end.i.i1415, label %land.lhs.true.i.i1410

land.lhs.true.i.i1410:                            ; preds = %ehcleanup360
  %refcount.i.i1408 = getelementptr inbounds %"struct.cv::UMatData", %"struct.cv::UMatData"* %430, i64 0, i32 3
  %431 = atomicrmw add i32* %refcount.i.i1408, i32 -1 acq_rel
  %cmp.i.i1409 = icmp eq i32 %431, 1
  br i1 %cmp.i.i1409, label %if.then.i.i1411, label %if.end.i.i1415

if.then.i.i1411:                                  ; preds = %land.lhs.true.i.i1410
  invoke void @_ZN2cv3Mat10deallocateEv(%"class.cv::Mat"* nonnull %section_p_out_count)
          to label %if.end.i.i1415 unwind label %terminate.lpad.i1429

if.end.i.i1415:                                   ; preds = %if.then.i.i1411, %land.lhs.true.i.i1410, %ehcleanup360
  store %"struct.cv::UMatData"* null, %"struct.cv::UMatData"** %u.i.i1356, align 8, !tbaa !76
  call void @llvm.memset.p0i8.i64(i8* nonnull %64, i8 0, i64 32, i32 8, i1 false)
  %432 = load i32, i32* %dims.i654, align 4, !tbaa !24
  %cmp48.i.i1414 = icmp sgt i32 %432, 0
  br i1 %cmp48.i.i1414, label %for.body.lr.ph.i.i1417, label %invoke.cont.i1426

for.body.lr.ph.i.i1417:                           ; preds = %if.end.i.i1415
  %433 = load i32*, i32** %p.i.i656, align 8, !tbaa !77
  br label %for.body.i.i1422

for.body.i.i1422:                                 ; preds = %for.body.i.i1422, %for.body.lr.ph.i.i1417
  %indvars.iv.i.i1418 = phi i64 [ 0, %for.body.lr.ph.i.i1417 ], [ %indvars.iv.next.i.i1420, %for.body.i.i1422 ]
  %arrayidx.i.i1419 = getelementptr inbounds i32, i32* %433, i64 %indvars.iv.i.i1418
  store i32 0, i32* %arrayidx.i.i1419, align 4, !tbaa !13
  %indvars.iv.next.i.i1420 = add nuw nsw i64 %indvars.iv.i.i1418, 1
  %434 = load i32, i32* %dims.i654, align 4, !tbaa !24
  %435 = sext i32 %434 to i64
  %cmp4.i.i1421 = icmp slt i64 %indvars.iv.next.i.i1420, %435
  br i1 %cmp4.i.i1421, label %for.body.i.i1422, label %invoke.cont.i1426

invoke.cont.i1426:                                ; preds = %for.body.i.i1422, %if.end.i.i1415
  %436 = load i64*, i64** %p.i3.i658, align 8, !tbaa !34
  %cmp.i1425 = icmp eq i64* %436, %arraydecay.i.i657
  br i1 %cmp.i1425, label %ehcleanup361, label %if.then.i1427

if.then.i1427:                                    ; preds = %invoke.cont.i1426
  %437 = bitcast i64* %436 to i8*
  invoke void @_ZN2cv8fastFreeEPv(i8* %437)
          to label %ehcleanup361 unwind label %terminate.lpad.i1429

terminate.lpad.i1429:                             ; preds = %if.then.i1427, %if.then.i.i1411
  %438 = landingpad { i8*, i32 }
          catch i8* null
  %439 = extractvalue { i8*, i32 } %438, 0
  call void @__clang_call_terminate(i8* %439)
  unreachable

ehcleanup361:                                     ; preds = %if.then.i1427, %invoke.cont.i1426
  call void @llvm.lifetime.end.p0i8(i64 96, i8* nonnull %39)
  %440 = load %"struct.cv::UMatData"*, %"struct.cv::UMatData"** %u.i.i1381, align 8, !tbaa !76
  %tobool.i.i1432 = icmp eq %"struct.cv::UMatData"* %440, null
  br i1 %tobool.i.i1432, label %if.end.i.i1440, label %land.lhs.true.i.i1435

land.lhs.true.i.i1435:                            ; preds = %ehcleanup361
  %refcount.i.i1433 = getelementptr inbounds %"struct.cv::UMatData", %"struct.cv::UMatData"* %440, i64 0, i32 3
  %441 = atomicrmw add i32* %refcount.i.i1433, i32 -1 acq_rel
  %cmp.i.i1434 = icmp eq i32 %441, 1
  br i1 %cmp.i.i1434, label %if.then.i.i1436, label %if.end.i.i1440

if.then.i.i1436:                                  ; preds = %land.lhs.true.i.i1435
  invoke void @_ZN2cv3Mat10deallocateEv(%"class.cv::Mat"* nonnull %section_p_out_sum)
          to label %if.end.i.i1440 unwind label %terminate.lpad.i1454

if.end.i.i1440:                                   ; preds = %if.then.i.i1436, %land.lhs.true.i.i1435, %ehcleanup361
  store %"struct.cv::UMatData"* null, %"struct.cv::UMatData"** %u.i.i1381, align 8, !tbaa !76
  call void @llvm.memset.p0i8.i64(i8* nonnull %68, i8 0, i64 32, i32 8, i1 false)
  %442 = load i32, i32* %dims.i634, align 4, !tbaa !24
  %cmp48.i.i1439 = icmp sgt i32 %442, 0
  br i1 %cmp48.i.i1439, label %for.body.lr.ph.i.i1442, label %invoke.cont.i1451

for.body.lr.ph.i.i1442:                           ; preds = %if.end.i.i1440
  %443 = load i32*, i32** %p.i.i636, align 8, !tbaa !77
  br label %for.body.i.i1447

for.body.i.i1447:                                 ; preds = %for.body.i.i1447, %for.body.lr.ph.i.i1442
  %indvars.iv.i.i1443 = phi i64 [ 0, %for.body.lr.ph.i.i1442 ], [ %indvars.iv.next.i.i1445, %for.body.i.i1447 ]
  %arrayidx.i.i1444 = getelementptr inbounds i32, i32* %443, i64 %indvars.iv.i.i1443
  store i32 0, i32* %arrayidx.i.i1444, align 4, !tbaa !13
  %indvars.iv.next.i.i1445 = add nuw nsw i64 %indvars.iv.i.i1443, 1
  %444 = load i32, i32* %dims.i634, align 4, !tbaa !24
  %445 = sext i32 %444 to i64
  %cmp4.i.i1446 = icmp slt i64 %indvars.iv.next.i.i1445, %445
  br i1 %cmp4.i.i1446, label %for.body.i.i1447, label %invoke.cont.i1451

invoke.cont.i1451:                                ; preds = %for.body.i.i1447, %if.end.i.i1440
  %446 = load i64*, i64** %p.i3.i638, align 8, !tbaa !34
  %cmp.i1450 = icmp eq i64* %446, %arraydecay.i.i637
  br i1 %cmp.i1450, label %ehcleanup363, label %if.then.i1452

if.then.i1452:                                    ; preds = %invoke.cont.i1451
  %447 = bitcast i64* %446 to i8*
  invoke void @_ZN2cv8fastFreeEPv(i8* %447)
          to label %ehcleanup363 unwind label %terminate.lpad.i1454

terminate.lpad.i1454:                             ; preds = %if.then.i1452, %if.then.i.i1436
  %448 = landingpad { i8*, i32 }
          catch i8* null
  %449 = extractvalue { i8*, i32 } %448, 0
  call void @__clang_call_terminate(i8* %449)
  unreachable

ehcleanup363:                                     ; preds = %if.then.i1452, %invoke.cont.i1451
  call void @llvm.lifetime.end.p0i8(i64 96, i8* nonnull %36)
  br label %ehcleanup411

for.cond.cleanup375:                              ; preds = %_ZNSsD2Ev.exit755, %for.cond.cleanup, %invoke.cont14
  %u.i.i1459 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %last_img, i64 0, i32 9
  %450 = load %"struct.cv::UMatData"*, %"struct.cv::UMatData"** %u.i.i1459, align 8, !tbaa !76
  %tobool.i.i1460 = icmp eq %"struct.cv::UMatData"* %450, null
  br i1 %tobool.i.i1460, label %if.end.i.i1468, label %land.lhs.true.i.i1463

land.lhs.true.i.i1463:                            ; preds = %for.cond.cleanup375
  %refcount.i.i1461 = getelementptr inbounds %"struct.cv::UMatData", %"struct.cv::UMatData"* %450, i64 0, i32 3
  %451 = atomicrmw add i32* %refcount.i.i1461, i32 -1 acq_rel
  %cmp.i.i1462 = icmp eq i32 %451, 1
  br i1 %cmp.i.i1462, label %if.then.i.i1464, label %if.end.i.i1468

if.then.i.i1464:                                  ; preds = %land.lhs.true.i.i1463
  invoke void @_ZN2cv3Mat10deallocateEv(%"class.cv::Mat"* nonnull %last_img)
          to label %if.end.i.i1468 unwind label %terminate.lpad.i1482

if.end.i.i1468:                                   ; preds = %if.then.i.i1464, %land.lhs.true.i.i1463, %for.cond.cleanup375
  store %"struct.cv::UMatData"* null, %"struct.cv::UMatData"** %u.i.i1459, align 8, !tbaa !76
  %data.i.i1465 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %last_img, i64 0, i32 4
  %452 = bitcast i8** %data.i.i1465 to i8*
  call void @llvm.memset.p0i8.i64(i8* nonnull %452, i8 0, i64 32, i32 8, i1 false)
  %453 = load i32, i32* %dims.i, align 4, !tbaa !24
  %cmp48.i.i1467 = icmp sgt i32 %453, 0
  br i1 %cmp48.i.i1467, label %for.body.lr.ph.i.i1470, label %invoke.cont.i1479

for.body.lr.ph.i.i1470:                           ; preds = %if.end.i.i1468
  %454 = load i32*, i32** %p.i.i603, align 8, !tbaa !77
  br label %for.body.i.i1475

for.body.i.i1475:                                 ; preds = %for.body.i.i1475, %for.body.lr.ph.i.i1470
  %indvars.iv.i.i1471 = phi i64 [ 0, %for.body.lr.ph.i.i1470 ], [ %indvars.iv.next.i.i1473, %for.body.i.i1475 ]
  %arrayidx.i.i1472 = getelementptr inbounds i32, i32* %454, i64 %indvars.iv.i.i1471
  store i32 0, i32* %arrayidx.i.i1472, align 4, !tbaa !13
  %indvars.iv.next.i.i1473 = add nuw nsw i64 %indvars.iv.i.i1471, 1
  %455 = load i32, i32* %dims.i, align 4, !tbaa !24
  %456 = sext i32 %455 to i64
  %cmp4.i.i1474 = icmp slt i64 %indvars.iv.next.i.i1473, %456
  br i1 %cmp4.i.i1474, label %for.body.i.i1475, label %invoke.cont.i1479

invoke.cont.i1479:                                ; preds = %for.body.i.i1475, %if.end.i.i1468
  %457 = load i64*, i64** %p.i3.i, align 8, !tbaa !34
  %cmp.i1478 = icmp eq i64* %457, %arraydecay.i.i
  br i1 %cmp.i1478, label %_ZN2cv3MatD2Ev.exit1483, label %if.then.i1480

if.then.i1480:                                    ; preds = %invoke.cont.i1479
  %458 = bitcast i64* %457 to i8*
  invoke void @_ZN2cv8fastFreeEPv(i8* %458)
          to label %_ZN2cv3MatD2Ev.exit1483 unwind label %terminate.lpad.i1482

terminate.lpad.i1482:                             ; preds = %if.then.i1480, %if.then.i.i1464
  %459 = landingpad { i8*, i32 }
          catch i8* null
  %460 = extractvalue { i8*, i32 } %459, 0
  call void @__clang_call_terminate(i8* %460)
  unreachable

_ZN2cv3MatD2Ev.exit1483:                          ; preds = %if.then.i1480, %invoke.cont.i1479
  call void @llvm.lifetime.end.p0i8(i64 96, i8* nonnull %31)
  %u.i.i963 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %img, i64 0, i32 9
  %461 = load %"struct.cv::UMatData"*, %"struct.cv::UMatData"** %u.i.i963, align 8, !tbaa !76
  %tobool.i.i964 = icmp eq %"struct.cv::UMatData"* %461, null
  br i1 %tobool.i.i964, label %if.end.i.i972, label %land.lhs.true.i.i967

land.lhs.true.i.i967:                             ; preds = %_ZN2cv3MatD2Ev.exit1483
  %refcount.i.i965 = getelementptr inbounds %"struct.cv::UMatData", %"struct.cv::UMatData"* %461, i64 0, i32 3
  %462 = atomicrmw add i32* %refcount.i.i965, i32 -1 acq_rel
  %cmp.i.i966 = icmp eq i32 %462, 1
  br i1 %cmp.i.i966, label %if.then.i.i968, label %if.end.i.i972

if.then.i.i968:                                   ; preds = %land.lhs.true.i.i967
  invoke void @_ZN2cv3Mat10deallocateEv(%"class.cv::Mat"* nonnull %img)
          to label %if.end.i.i972 unwind label %terminate.lpad.i986

if.end.i.i972:                                    ; preds = %if.then.i.i968, %land.lhs.true.i.i967, %_ZN2cv3MatD2Ev.exit1483
  store %"struct.cv::UMatData"* null, %"struct.cv::UMatData"** %u.i.i963, align 8, !tbaa !76
  %data.i.i969 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %img, i64 0, i32 4
  %dims.i.i970 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %img, i64 0, i32 1
  %463 = bitcast i8** %data.i.i969 to i8*
  call void @llvm.memset.p0i8.i64(i8* nonnull %463, i8 0, i64 32, i32 8, i1 false)
  %464 = load i32, i32* %dims.i.i970, align 4, !tbaa !24
  %cmp48.i.i971 = icmp sgt i32 %464, 0
  br i1 %cmp48.i.i971, label %for.body.lr.ph.i.i974, label %invoke.cont.i983

for.body.lr.ph.i.i974:                            ; preds = %if.end.i.i972
  %p.i.i973 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %img, i64 0, i32 10, i32 0
  %465 = load i32*, i32** %p.i.i973, align 8, !tbaa !77
  br label %for.body.i.i979

for.body.i.i979:                                  ; preds = %for.body.i.i979, %for.body.lr.ph.i.i974
  %indvars.iv.i.i975 = phi i64 [ 0, %for.body.lr.ph.i.i974 ], [ %indvars.iv.next.i.i977, %for.body.i.i979 ]
  %arrayidx.i.i976 = getelementptr inbounds i32, i32* %465, i64 %indvars.iv.i.i975
  store i32 0, i32* %arrayidx.i.i976, align 4, !tbaa !13
  %indvars.iv.next.i.i977 = add nuw nsw i64 %indvars.iv.i.i975, 1
  %466 = load i32, i32* %dims.i.i970, align 4, !tbaa !24
  %467 = sext i32 %466 to i64
  %cmp4.i.i978 = icmp slt i64 %indvars.iv.next.i.i977, %467
  br i1 %cmp4.i.i978, label %for.body.i.i979, label %invoke.cont.i983

invoke.cont.i983:                                 ; preds = %for.body.i.i979, %if.end.i.i972
  %p.i980 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %img, i64 0, i32 11, i32 0
  %468 = load i64*, i64** %p.i980, align 8, !tbaa !34
  %arraydecay.i981 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %img, i64 0, i32 11, i32 1, i64 0
  %cmp.i982 = icmp eq i64* %468, %arraydecay.i981
  br i1 %cmp.i982, label %_ZN2cv3MatD2Ev.exit987, label %if.then.i984

if.then.i984:                                     ; preds = %invoke.cont.i983
  %469 = bitcast i64* %468 to i8*
  invoke void @_ZN2cv8fastFreeEPv(i8* %469)
          to label %_ZN2cv3MatD2Ev.exit987 unwind label %terminate.lpad.i986

terminate.lpad.i986:                              ; preds = %if.then.i984, %if.then.i.i968
  %470 = landingpad { i8*, i32 }
          catch i8* null
  %471 = extractvalue { i8*, i32 } %470, 0
  call void @__clang_call_terminate(i8* %471)
  unreachable

_ZN2cv3MatD2Ev.exit987:                           ; preds = %if.then.i984, %invoke.cont.i983
  call void @llvm.lifetime.end.p0i8(i64 96, i8* nonnull %26)
  %u.i.i917 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %next_section_img, i64 0, i32 9
  %472 = load %"struct.cv::UMatData"*, %"struct.cv::UMatData"** %u.i.i917, align 8, !tbaa !76
  %tobool.i.i918 = icmp eq %"struct.cv::UMatData"* %472, null
  br i1 %tobool.i.i918, label %if.end.i.i926, label %land.lhs.true.i.i921

land.lhs.true.i.i921:                             ; preds = %_ZN2cv3MatD2Ev.exit987
  %refcount.i.i919 = getelementptr inbounds %"struct.cv::UMatData", %"struct.cv::UMatData"* %472, i64 0, i32 3
  %473 = atomicrmw add i32* %refcount.i.i919, i32 -1 acq_rel
  %cmp.i.i920 = icmp eq i32 %473, 1
  br i1 %cmp.i.i920, label %if.then.i.i922, label %if.end.i.i926

if.then.i.i922:                                   ; preds = %land.lhs.true.i.i921
  invoke void @_ZN2cv3Mat10deallocateEv(%"class.cv::Mat"* nonnull %next_section_img)
          to label %if.end.i.i926 unwind label %terminate.lpad.i940

if.end.i.i926:                                    ; preds = %if.then.i.i922, %land.lhs.true.i.i921, %_ZN2cv3MatD2Ev.exit987
  store %"struct.cv::UMatData"* null, %"struct.cv::UMatData"** %u.i.i917, align 8, !tbaa !76
  %data.i.i923 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %next_section_img, i64 0, i32 4
  %dims.i.i924 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %next_section_img, i64 0, i32 1
  %474 = bitcast i8** %data.i.i923 to i8*
  call void @llvm.memset.p0i8.i64(i8* nonnull %474, i8 0, i64 32, i32 16, i1 false)
  %475 = load i32, i32* %dims.i.i924, align 4, !tbaa !24
  %cmp48.i.i925 = icmp sgt i32 %475, 0
  br i1 %cmp48.i.i925, label %for.body.lr.ph.i.i928, label %invoke.cont.i937

for.body.lr.ph.i.i928:                            ; preds = %if.end.i.i926
  %p.i.i927 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %next_section_img, i64 0, i32 10, i32 0
  %476 = load i32*, i32** %p.i.i927, align 16, !tbaa !77
  br label %for.body.i.i933

for.body.i.i933:                                  ; preds = %for.body.i.i933, %for.body.lr.ph.i.i928
  %indvars.iv.i.i929 = phi i64 [ 0, %for.body.lr.ph.i.i928 ], [ %indvars.iv.next.i.i931, %for.body.i.i933 ]
  %arrayidx.i.i930 = getelementptr inbounds i32, i32* %476, i64 %indvars.iv.i.i929
  store i32 0, i32* %arrayidx.i.i930, align 4, !tbaa !13
  %indvars.iv.next.i.i931 = add nuw nsw i64 %indvars.iv.i.i929, 1
  %477 = load i32, i32* %dims.i.i924, align 4, !tbaa !24
  %478 = sext i32 %477 to i64
  %cmp4.i.i932 = icmp slt i64 %indvars.iv.next.i.i931, %478
  br i1 %cmp4.i.i932, label %for.body.i.i933, label %invoke.cont.i937

invoke.cont.i937:                                 ; preds = %for.body.i.i933, %if.end.i.i926
  %p.i934 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %next_section_img, i64 0, i32 11, i32 0
  %479 = load i64*, i64** %p.i934, align 8, !tbaa !34
  %arraydecay.i935 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %next_section_img, i64 0, i32 11, i32 1, i64 0
  %cmp.i936 = icmp eq i64* %479, %arraydecay.i935
  br i1 %cmp.i936, label %_ZN2cv3MatD2Ev.exit941, label %if.then.i938

if.then.i938:                                     ; preds = %invoke.cont.i937
  %480 = bitcast i64* %479 to i8*
  invoke void @_ZN2cv8fastFreeEPv(i8* %480)
          to label %_ZN2cv3MatD2Ev.exit941 unwind label %terminate.lpad.i940

terminate.lpad.i940:                              ; preds = %if.then.i938, %if.then.i.i922
  %481 = landingpad { i8*, i32 }
          catch i8* null
  %482 = extractvalue { i8*, i32 } %481, 0
  call void @__clang_call_terminate(i8* %482)
  unreachable

_ZN2cv3MatD2Ev.exit941:                           ; preds = %if.then.i938, %invoke.cont.i937
  call void @llvm.lifetime.end.p0i8(i64 96, i8* nonnull %23)
  br label %cleanup.cont

invoke.cont383:                                   ; preds = %_ZNSsD2Ev.exit755, %invoke.cont383.lr.ph
  %.in = phi i64 [ %428, %invoke.cont383.lr.ph ], [ %527, %_ZNSsD2Ev.exit755 ]
  %indvars.iv = phi i64 [ 0, %invoke.cont383.lr.ph ], [ %indvars.iv.next, %_ZNSsD2Ev.exit755 ]
  %483 = inttoptr i64 %.in to %"class.tfk::Section"**
  %add.ptr.i916 = getelementptr inbounds %"class.tfk::Section"*, %"class.tfk::Section"** %483, i64 %indvars.iv
  %484 = load %"class.tfk::Section"*, %"class.tfk::Section"** %add.ptr.i916, align 8, !tbaa !15
  %485 = load <4 x i32>, <4 x i32>* %91, align 4, !tbaa !7
  store <4 x i32> %485, <4 x i32>* %agg.tmp381, align 16, !tbaa !7
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %82)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %83)
  invoke void @_ZNSsC1ERKSs(%"class.std::basic_string"* nonnull %ref.tmp386, %"class.std::basic_string"* nonnull dereferenceable(8) %filename_prefix)
          to label %.noexc880 unwind label %lpad387

.noexc880:                                        ; preds = %invoke.cont383
  %call2.i2.i = invoke dereferenceable(8) %"class.std::basic_string"* @_ZNSs6appendEPKcm(%"class.std::basic_string"* nonnull %ref.tmp386, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.4, i64 0, i64 0), i64 1)
          to label %invoke.cont388 unwind label %lpad.i

lpad.i:                                           ; preds = %.noexc880
  %486 = landingpad { i8*, i32 }
          cleanup
  %487 = load i8*, i8** %_M_p.i.i.i.i, align 8, !tbaa !9, !alias.scope !79
  %arrayidx.i.i.i878 = getelementptr inbounds i8, i8* %487, i64 -24
  %488 = bitcast i8* %arrayidx.i.i.i878 to %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"*
  %489 = getelementptr inbounds %"class.std::allocator", %"class.std::allocator"* %ref.tmp.i.i, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %489), !noalias !79
  %cmp.i.i.i = icmp eq i8* %arrayidx.i.i.i878, bitcast ([0 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE to i8*)
  br i1 %cmp.i.i.i, label %_ZNSsD2Ev.exit.i, label %if.then.i.i.i879, !prof !12

if.then.i.i.i879:                                 ; preds = %lpad.i
  %_M_refcount.i.i.i = getelementptr inbounds i8, i8* %487, i64 -8
  %490 = bitcast i8* %_M_refcount.i.i.i to i32*
  br i1 icmp ne (i8* bitcast (i32 (i32*, void (i8*)*)* @__pthread_key_create to i8*), i8* null), label %if.then.i.i.i.i, label %if.else.i.i.i.i

if.then.i.i.i.i:                                  ; preds = %if.then.i.i.i879
  %491 = atomicrmw volatile add i32* %490, i32 -1 acq_rel
  br label %invoke.cont.i.i.i

if.else.i.i.i.i:                                  ; preds = %if.then.i.i.i879
  %492 = load i32, i32* %490, align 4, !tbaa !13
  %add.i.i.i.i.i = add nsw i32 %492, -1
  store i32 %add.i.i.i.i.i, i32* %490, align 4, !tbaa !13
  br label %invoke.cont.i.i.i

invoke.cont.i.i.i:                                ; preds = %if.else.i.i.i.i, %if.then.i.i.i.i
  %retval.0.i.i.i.i = phi i32 [ %491, %if.then.i.i.i.i ], [ %492, %if.else.i.i.i.i ]
  %cmp3.i.i.i = icmp slt i32 %retval.0.i.i.i.i, 1
  br i1 %cmp3.i.i.i, label %if.then4.i.i.i, label %_ZNSsD2Ev.exit.i

if.then4.i.i.i:                                   ; preds = %invoke.cont.i.i.i
  call void @_ZNSs4_Rep10_M_destroyERKSaIcE(%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"* nonnull %488, %"class.std::allocator"* nonnull dereferenceable(1) %ref.tmp.i.i)
  br label %_ZNSsD2Ev.exit.i

_ZNSsD2Ev.exit.i:                                 ; preds = %if.then4.i.i.i, %invoke.cont.i.i.i, %lpad.i
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %489), !noalias !79
  br label %lpad387.body

invoke.cont388:                                   ; preds = %.noexc880
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %84)
  %real_section_id390 = getelementptr inbounds %"class.tfk::Section", %"class.tfk::Section"* %484, i64 0, i32 1
  %493 = load i32, i32* %real_section_id390, align 4, !tbaa !42
  invoke void (%"class.std::basic_string"*, i32 (i8*, i64, i8*, %struct.__va_list_tag*)*, i64, i8*, ...) @_ZN9__gnu_cxx12__to_xstringISscEET_PFiPT0_mPKS2_P13__va_list_tagEmS5_z(%"class.std::basic_string"* nonnull sret %ref.tmp389, i32 (i8*, i64, i8*, %struct.__va_list_tag*)* nonnull @vsnprintf, i64 16, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.16, i64 0, i64 0), i32 %493)
          to label %invoke.cont392 unwind label %lpad391

invoke.cont392:                                   ; preds = %invoke.cont388
  %494 = load i8*, i8** %_M_p.i.i.i.i, align 8, !tbaa !9, !noalias !82
  %arrayidx.i.i.i = getelementptr inbounds i8, i8* %494, i64 -24
  %_M_length.i.i = bitcast i8* %arrayidx.i.i.i to i64*
  %495 = load i64, i64* %_M_length.i.i, align 8, !tbaa !60, !noalias !82
  %496 = load i8*, i8** %_M_p.i.i.i17.i, align 8, !tbaa !9, !noalias !82
  %arrayidx.i.i18.i = getelementptr inbounds i8, i8* %496, i64 -24
  %_M_length.i19.i = bitcast i8* %arrayidx.i.i18.i to i64*
  %497 = load i64, i64* %_M_length.i19.i, align 8, !tbaa !60, !noalias !82
  %add.i = add i64 %497, %495
  %_M_capacity.i22.i = getelementptr inbounds i8, i8* %494, i64 -16
  %498 = bitcast i8* %_M_capacity.i22.i to i64*
  %499 = load i64, i64* %498, align 8, !tbaa !62, !noalias !82
  %cmp.i849 = icmp ugt i64 %add.i, %499
  br i1 %cmp.i849, label %land.rhs.i, label %cond.false.i

land.rhs.i:                                       ; preds = %invoke.cont392
  %_M_capacity.i.i = getelementptr inbounds i8, i8* %496, i64 -16
  %500 = bitcast i8* %_M_capacity.i.i to i64*
  %501 = load i64, i64* %500, align 8, !tbaa !62, !noalias !82
  %cmp4.i850 = icmp ugt i64 %add.i, %501
  br i1 %cmp4.i850, label %cond.false.i, label %cond.true.i

cond.true.i:                                      ; preds = %land.rhs.i
  %call4.i.i.i852 = invoke dereferenceable(8) %"class.std::basic_string"* @_ZNSs6insertEmPKcm(%"class.std::basic_string"* nonnull %ref.tmp389, i64 0, i8* %494, i64 %495)
          to label %invoke.cont394 unwind label %lpad393

cond.false.i:                                     ; preds = %land.rhs.i, %invoke.cont392
  %call7.i853 = invoke dereferenceable(8) %"class.std::basic_string"* @_ZNSs6appendERKSs(%"class.std::basic_string"* nonnull %ref.tmp386, %"class.std::basic_string"* nonnull dereferenceable(8) %ref.tmp389)
          to label %invoke.cont394 unwind label %lpad393

invoke.cont394:                                   ; preds = %cond.false.i, %cond.true.i
  %cond-lvalue.i = phi %"class.std::basic_string"* [ %call4.i.i.i852, %cond.true.i ], [ %call7.i853, %cond.false.i ]
  %502 = bitcast %"class.std::basic_string"* %cond-lvalue.i to i64*
  %503 = load i64, i64* %502, align 8, !tbaa !63, !noalias !82
  store i64 %503, i64* %85, align 8, !tbaa !63, !alias.scope !82
  %_M_p.i.i.i851 = getelementptr inbounds %"class.std::basic_string", %"class.std::basic_string"* %cond-lvalue.i, i64 0, i32 0, i32 0
  store i8* bitcast (i64* getelementptr inbounds ([0 x i64], [0 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE, i64 0, i64 3) to i8*), i8** %_M_p.i.i.i851, align 8, !tbaa !9, !noalias !82
  %call2.i.i834 = invoke dereferenceable(8) %"class.std::basic_string"* @_ZNSs6appendEPKcm(%"class.std::basic_string"* nonnull %ref.tmp385, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.5, i64 0, i64 0), i64 4)
          to label %invoke.cont396 unwind label %lpad395

invoke.cont396:                                   ; preds = %invoke.cont394
  %504 = bitcast %"class.std::basic_string"* %call2.i.i834 to i64*
  %505 = load i64, i64* %504, align 8, !tbaa !63, !noalias !85
  store i64 %505, i64* %86, align 8, !tbaa !63, !alias.scope !85
  %_M_p.i.i.i833 = getelementptr inbounds %"class.std::basic_string", %"class.std::basic_string"* %call2.i.i834, i64 0, i32 0, i32 0
  store i8* bitcast (i64* getelementptr inbounds ([0 x i64], [0 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE, i64 0, i64 3) to i8*), i8** %_M_p.i.i.i833, align 8, !tbaa !9, !noalias !85
  %506 = inttoptr i64 %505 to i8*
  invoke void @_ZN3tfk6Render6renderEPNS_7SectionESt4pairIN2cv6Point_IfEES6_ENS_10ResolutionESs(%"class.tfk::Render"* %this, %"class.tfk::Section"* nonnull %484, %"struct.std::pair"* nonnull %tmpcast1940, i32 %resolution, %"class.std::basic_string"* nonnull %agg.tmp384)
          to label %invoke.cont398 unwind label %lpad397

invoke.cont398:                                   ; preds = %invoke.cont396
  %arrayidx.i.i807 = getelementptr inbounds i8, i8* %506, i64 -24
  %507 = bitcast i8* %arrayidx.i.i807 to %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %87)
  %cmp.i.i808 = icmp eq i8* %arrayidx.i.i807, bitcast ([0 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE to i8*)
  br i1 %cmp.i.i808, label %_ZNSsD2Ev.exit818, label %if.then.i.i810, !prof !12

if.then.i.i810:                                   ; preds = %invoke.cont398
  %_M_refcount.i.i809 = getelementptr inbounds i8, i8* %506, i64 -8
  %508 = bitcast i8* %_M_refcount.i.i809 to i32*
  br i1 icmp ne (i8* bitcast (i32 (i32*, void (i8*)*)* @__pthread_key_create to i8*), i8* null), label %if.then.i.i.i811, label %if.else.i.i.i813

if.then.i.i.i811:                                 ; preds = %if.then.i.i810
  %509 = atomicrmw volatile add i32* %508, i32 -1 acq_rel
  br label %invoke.cont.i.i816

if.else.i.i.i813:                                 ; preds = %if.then.i.i810
  %510 = load i32, i32* %508, align 4, !tbaa !13
  %add.i.i.i.i812 = add nsw i32 %510, -1
  store i32 %add.i.i.i.i812, i32* %508, align 4, !tbaa !13
  br label %invoke.cont.i.i816

invoke.cont.i.i816:                               ; preds = %if.else.i.i.i813, %if.then.i.i.i811
  %retval.0.i.i.i814 = phi i32 [ %509, %if.then.i.i.i811 ], [ %510, %if.else.i.i.i813 ]
  %cmp3.i.i815 = icmp slt i32 %retval.0.i.i.i814, 1
  br i1 %cmp3.i.i815, label %if.then4.i.i817, label %_ZNSsD2Ev.exit818

if.then4.i.i817:                                  ; preds = %invoke.cont.i.i816
  call void @_ZNSs4_Rep10_M_destroyERKSaIcE(%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"* nonnull %507, %"class.std::allocator"* nonnull dereferenceable(1) %ref.tmp.i805)
  br label %_ZNSsD2Ev.exit818

_ZNSsD2Ev.exit818:                                ; preds = %if.then4.i.i817, %invoke.cont.i.i816, %invoke.cont398
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %87)
  %511 = load i8*, i8** %_M_p.i.i.i785, align 8, !tbaa !9
  %arrayidx.i.i786 = getelementptr inbounds i8, i8* %511, i64 -24
  %512 = bitcast i8* %arrayidx.i.i786 to %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %88)
  %cmp.i.i787 = icmp eq i8* %arrayidx.i.i786, bitcast ([0 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE to i8*)
  br i1 %cmp.i.i787, label %_ZNSsD2Ev.exit797, label %if.then.i.i789, !prof !12

if.then.i.i789:                                   ; preds = %_ZNSsD2Ev.exit818
  %_M_refcount.i.i788 = getelementptr inbounds i8, i8* %511, i64 -8
  %513 = bitcast i8* %_M_refcount.i.i788 to i32*
  br i1 icmp ne (i8* bitcast (i32 (i32*, void (i8*)*)* @__pthread_key_create to i8*), i8* null), label %if.then.i.i.i790, label %if.else.i.i.i792

if.then.i.i.i790:                                 ; preds = %if.then.i.i789
  %514 = atomicrmw volatile add i32* %513, i32 -1 acq_rel
  br label %invoke.cont.i.i795

if.else.i.i.i792:                                 ; preds = %if.then.i.i789
  %515 = load i32, i32* %513, align 4, !tbaa !13
  %add.i.i.i.i791 = add nsw i32 %515, -1
  store i32 %add.i.i.i.i791, i32* %513, align 4, !tbaa !13
  br label %invoke.cont.i.i795

invoke.cont.i.i795:                               ; preds = %if.else.i.i.i792, %if.then.i.i.i790
  %retval.0.i.i.i793 = phi i32 [ %514, %if.then.i.i.i790 ], [ %515, %if.else.i.i.i792 ]
  %cmp3.i.i794 = icmp slt i32 %retval.0.i.i.i793, 1
  br i1 %cmp3.i.i794, label %if.then4.i.i796, label %_ZNSsD2Ev.exit797

if.then4.i.i796:                                  ; preds = %invoke.cont.i.i795
  call void @_ZNSs4_Rep10_M_destroyERKSaIcE(%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"* nonnull %512, %"class.std::allocator"* nonnull dereferenceable(1) %ref.tmp.i784)
  br label %_ZNSsD2Ev.exit797

_ZNSsD2Ev.exit797:                                ; preds = %if.then4.i.i796, %invoke.cont.i.i795, %_ZNSsD2Ev.exit818
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %88)
  %516 = load i8*, i8** %_M_p.i.i.i17.i, align 8, !tbaa !9
  %arrayidx.i.i765 = getelementptr inbounds i8, i8* %516, i64 -24
  %517 = bitcast i8* %arrayidx.i.i765 to %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %89)
  %cmp.i.i766 = icmp eq i8* %arrayidx.i.i765, bitcast ([0 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE to i8*)
  br i1 %cmp.i.i766, label %_ZNSsD2Ev.exit776, label %if.then.i.i768, !prof !12

if.then.i.i768:                                   ; preds = %_ZNSsD2Ev.exit797
  %_M_refcount.i.i767 = getelementptr inbounds i8, i8* %516, i64 -8
  %518 = bitcast i8* %_M_refcount.i.i767 to i32*
  br i1 icmp ne (i8* bitcast (i32 (i32*, void (i8*)*)* @__pthread_key_create to i8*), i8* null), label %if.then.i.i.i769, label %if.else.i.i.i771

if.then.i.i.i769:                                 ; preds = %if.then.i.i768
  %519 = atomicrmw volatile add i32* %518, i32 -1 acq_rel
  br label %invoke.cont.i.i774

if.else.i.i.i771:                                 ; preds = %if.then.i.i768
  %520 = load i32, i32* %518, align 4, !tbaa !13
  %add.i.i.i.i770 = add nsw i32 %520, -1
  store i32 %add.i.i.i.i770, i32* %518, align 4, !tbaa !13
  br label %invoke.cont.i.i774

invoke.cont.i.i774:                               ; preds = %if.else.i.i.i771, %if.then.i.i.i769
  %retval.0.i.i.i772 = phi i32 [ %519, %if.then.i.i.i769 ], [ %520, %if.else.i.i.i771 ]
  %cmp3.i.i773 = icmp slt i32 %retval.0.i.i.i772, 1
  br i1 %cmp3.i.i773, label %if.then4.i.i775, label %_ZNSsD2Ev.exit776

if.then4.i.i775:                                  ; preds = %invoke.cont.i.i774
  call void @_ZNSs4_Rep10_M_destroyERKSaIcE(%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"* nonnull %517, %"class.std::allocator"* nonnull dereferenceable(1) %ref.tmp.i763)
  br label %_ZNSsD2Ev.exit776

_ZNSsD2Ev.exit776:                                ; preds = %if.then4.i.i775, %invoke.cont.i.i774, %_ZNSsD2Ev.exit797
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %89)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %84)
  %521 = load i8*, i8** %_M_p.i.i.i.i, align 8, !tbaa !9
  %arrayidx.i.i744 = getelementptr inbounds i8, i8* %521, i64 -24
  %522 = bitcast i8* %arrayidx.i.i744 to %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %90)
  %cmp.i.i745 = icmp eq i8* %arrayidx.i.i744, bitcast ([0 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE to i8*)
  br i1 %cmp.i.i745, label %_ZNSsD2Ev.exit755, label %if.then.i.i747, !prof !12

if.then.i.i747:                                   ; preds = %_ZNSsD2Ev.exit776
  %_M_refcount.i.i746 = getelementptr inbounds i8, i8* %521, i64 -8
  %523 = bitcast i8* %_M_refcount.i.i746 to i32*
  br i1 icmp ne (i8* bitcast (i32 (i32*, void (i8*)*)* @__pthread_key_create to i8*), i8* null), label %if.then.i.i.i748, label %if.else.i.i.i750

if.then.i.i.i748:                                 ; preds = %if.then.i.i747
  %524 = atomicrmw volatile add i32* %523, i32 -1 acq_rel
  br label %invoke.cont.i.i753

if.else.i.i.i750:                                 ; preds = %if.then.i.i747
  %525 = load i32, i32* %523, align 4, !tbaa !13
  %add.i.i.i.i749 = add nsw i32 %525, -1
  store i32 %add.i.i.i.i749, i32* %523, align 4, !tbaa !13
  br label %invoke.cont.i.i753

invoke.cont.i.i753:                               ; preds = %if.else.i.i.i750, %if.then.i.i.i748
  %retval.0.i.i.i751 = phi i32 [ %524, %if.then.i.i.i748 ], [ %525, %if.else.i.i.i750 ]
  %cmp3.i.i752 = icmp slt i32 %retval.0.i.i.i751, 1
  br i1 %cmp3.i.i752, label %if.then4.i.i754, label %_ZNSsD2Ev.exit755

if.then4.i.i754:                                  ; preds = %invoke.cont.i.i753
  call void @_ZNSs4_Rep10_M_destroyERKSaIcE(%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"* nonnull %522, %"class.std::allocator"* nonnull dereferenceable(1) %ref.tmp.i742)
  br label %_ZNSsD2Ev.exit755

_ZNSsD2Ev.exit755:                                ; preds = %if.then4.i.i754, %invoke.cont.i.i753, %_ZNSsD2Ev.exit776
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %90)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %83)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %82)
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %526 = load i64, i64* %0, align 8, !tbaa !0
  %527 = load i64, i64* %2, align 8, !tbaa !6
  %sub.ptr.sub.i1457 = sub i64 %526, %527
  %sub.ptr.div.i1458 = ashr exact i64 %sub.ptr.sub.i1457, 3
  %cmp374 = icmp ugt i64 %sub.ptr.div.i1458, %indvars.iv.next
  br i1 %cmp374, label %invoke.cont383, label %for.cond.cleanup375

lpad387:                                          ; preds = %invoke.cont383
  %528 = landingpad { i8*, i32 }
          cleanup
  br label %lpad387.body

lpad387.body:                                     ; preds = %lpad387, %_ZNSsD2Ev.exit.i
  %eh.lpad-body = phi { i8*, i32 } [ %528, %lpad387 ], [ %486, %_ZNSsD2Ev.exit.i ]
  %529 = extractvalue { i8*, i32 } %eh.lpad-body, 0
  %530 = extractvalue { i8*, i32 } %eh.lpad-body, 1
  br label %ehcleanup404

lpad391:                                          ; preds = %invoke.cont388
  %531 = landingpad { i8*, i32 }
          cleanup
  %532 = extractvalue { i8*, i32 } %531, 0
  %533 = extractvalue { i8*, i32 } %531, 1
  br label %ehcleanup402

lpad393:                                          ; preds = %cond.false.i, %cond.true.i
  %534 = landingpad { i8*, i32 }
          cleanup
  %535 = extractvalue { i8*, i32 } %534, 0
  %536 = extractvalue { i8*, i32 } %534, 1
  br label %ehcleanup401

lpad395:                                          ; preds = %invoke.cont394
  %537 = landingpad { i8*, i32 }
          cleanup
  %538 = extractvalue { i8*, i32 } %537, 0
  %539 = extractvalue { i8*, i32 } %537, 1
  br label %ehcleanup400

lpad397:                                          ; preds = %invoke.cont396
  %540 = landingpad { i8*, i32 }
          cleanup
  %541 = inttoptr i64 %505 to i8*
  %542 = extractvalue { i8*, i32 } %540, 0
  %543 = extractvalue { i8*, i32 } %540, 1
  %arrayidx.i.i723 = getelementptr inbounds i8, i8* %541, i64 -24
  %544 = bitcast i8* %arrayidx.i.i723 to %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"*
  %545 = getelementptr inbounds %"class.std::allocator", %"class.std::allocator"* %ref.tmp.i721, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %545)
  %cmp.i.i724 = icmp eq i8* %arrayidx.i.i723, bitcast ([0 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE to i8*)
  br i1 %cmp.i.i724, label %_ZNSsD2Ev.exit734, label %if.then.i.i726, !prof !12

if.then.i.i726:                                   ; preds = %lpad397
  %_M_refcount.i.i725 = getelementptr inbounds i8, i8* %541, i64 -8
  %546 = bitcast i8* %_M_refcount.i.i725 to i32*
  br i1 icmp ne (i8* bitcast (i32 (i32*, void (i8*)*)* @__pthread_key_create to i8*), i8* null), label %if.then.i.i.i727, label %if.else.i.i.i729

if.then.i.i.i727:                                 ; preds = %if.then.i.i726
  %547 = atomicrmw volatile add i32* %546, i32 -1 acq_rel
  br label %invoke.cont.i.i732

if.else.i.i.i729:                                 ; preds = %if.then.i.i726
  %548 = load i32, i32* %546, align 4, !tbaa !13
  %add.i.i.i.i728 = add nsw i32 %548, -1
  store i32 %add.i.i.i.i728, i32* %546, align 4, !tbaa !13
  br label %invoke.cont.i.i732

invoke.cont.i.i732:                               ; preds = %if.else.i.i.i729, %if.then.i.i.i727
  %retval.0.i.i.i730 = phi i32 [ %547, %if.then.i.i.i727 ], [ %548, %if.else.i.i.i729 ]
  %cmp3.i.i731 = icmp slt i32 %retval.0.i.i.i730, 1
  br i1 %cmp3.i.i731, label %if.then4.i.i733, label %_ZNSsD2Ev.exit734

if.then4.i.i733:                                  ; preds = %invoke.cont.i.i732
  call void @_ZNSs4_Rep10_M_destroyERKSaIcE(%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"* nonnull %544, %"class.std::allocator"* nonnull dereferenceable(1) %ref.tmp.i721)
  br label %_ZNSsD2Ev.exit734

_ZNSsD2Ev.exit734:                                ; preds = %if.then4.i.i733, %invoke.cont.i.i732, %lpad397
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %545)
  br label %ehcleanup400

ehcleanup400:                                     ; preds = %_ZNSsD2Ev.exit734, %lpad395
  %ehselector.slot.10 = phi i32 [ %543, %_ZNSsD2Ev.exit734 ], [ %539, %lpad395 ]
  %exn.slot.10 = phi i8* [ %542, %_ZNSsD2Ev.exit734 ], [ %538, %lpad395 ]
  %549 = load i8*, i8** %_M_p.i.i.i785, align 8, !tbaa !9
  %arrayidx.i.i702 = getelementptr inbounds i8, i8* %549, i64 -24
  %550 = bitcast i8* %arrayidx.i.i702 to %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"*
  %551 = getelementptr inbounds %"class.std::allocator", %"class.std::allocator"* %ref.tmp.i700, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %551)
  %cmp.i.i703 = icmp eq i8* %arrayidx.i.i702, bitcast ([0 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE to i8*)
  br i1 %cmp.i.i703, label %_ZNSsD2Ev.exit713, label %if.then.i.i705, !prof !12

if.then.i.i705:                                   ; preds = %ehcleanup400
  %_M_refcount.i.i704 = getelementptr inbounds i8, i8* %549, i64 -8
  %552 = bitcast i8* %_M_refcount.i.i704 to i32*
  br i1 icmp ne (i8* bitcast (i32 (i32*, void (i8*)*)* @__pthread_key_create to i8*), i8* null), label %if.then.i.i.i706, label %if.else.i.i.i708

if.then.i.i.i706:                                 ; preds = %if.then.i.i705
  %553 = atomicrmw volatile add i32* %552, i32 -1 acq_rel
  br label %invoke.cont.i.i711

if.else.i.i.i708:                                 ; preds = %if.then.i.i705
  %554 = load i32, i32* %552, align 4, !tbaa !13
  %add.i.i.i.i707 = add nsw i32 %554, -1
  store i32 %add.i.i.i.i707, i32* %552, align 4, !tbaa !13
  br label %invoke.cont.i.i711

invoke.cont.i.i711:                               ; preds = %if.else.i.i.i708, %if.then.i.i.i706
  %retval.0.i.i.i709 = phi i32 [ %553, %if.then.i.i.i706 ], [ %554, %if.else.i.i.i708 ]
  %cmp3.i.i710 = icmp slt i32 %retval.0.i.i.i709, 1
  br i1 %cmp3.i.i710, label %if.then4.i.i712, label %_ZNSsD2Ev.exit713

if.then4.i.i712:                                  ; preds = %invoke.cont.i.i711
  call void @_ZNSs4_Rep10_M_destroyERKSaIcE(%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"* nonnull %550, %"class.std::allocator"* nonnull dereferenceable(1) %ref.tmp.i700)
  br label %_ZNSsD2Ev.exit713

_ZNSsD2Ev.exit713:                                ; preds = %if.then4.i.i712, %invoke.cont.i.i711, %ehcleanup400
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %551)
  br label %ehcleanup401

ehcleanup401:                                     ; preds = %_ZNSsD2Ev.exit713, %lpad393
  %ehselector.slot.11 = phi i32 [ %ehselector.slot.10, %_ZNSsD2Ev.exit713 ], [ %536, %lpad393 ]
  %exn.slot.11 = phi i8* [ %exn.slot.10, %_ZNSsD2Ev.exit713 ], [ %535, %lpad393 ]
  %555 = load i8*, i8** %_M_p.i.i.i17.i, align 8, !tbaa !9
  %arrayidx.i.i661 = getelementptr inbounds i8, i8* %555, i64 -24
  %556 = bitcast i8* %arrayidx.i.i661 to %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"*
  %557 = getelementptr inbounds %"class.std::allocator", %"class.std::allocator"* %ref.tmp.i659, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %557)
  %cmp.i.i662 = icmp eq i8* %arrayidx.i.i661, bitcast ([0 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE to i8*)
  br i1 %cmp.i.i662, label %_ZNSsD2Ev.exit672, label %if.then.i.i664, !prof !12

if.then.i.i664:                                   ; preds = %ehcleanup401
  %_M_refcount.i.i663 = getelementptr inbounds i8, i8* %555, i64 -8
  %558 = bitcast i8* %_M_refcount.i.i663 to i32*
  br i1 icmp ne (i8* bitcast (i32 (i32*, void (i8*)*)* @__pthread_key_create to i8*), i8* null), label %if.then.i.i.i665, label %if.else.i.i.i667

if.then.i.i.i665:                                 ; preds = %if.then.i.i664
  %559 = atomicrmw volatile add i32* %558, i32 -1 acq_rel
  br label %invoke.cont.i.i670

if.else.i.i.i667:                                 ; preds = %if.then.i.i664
  %560 = load i32, i32* %558, align 4, !tbaa !13
  %add.i.i.i.i666 = add nsw i32 %560, -1
  store i32 %add.i.i.i.i666, i32* %558, align 4, !tbaa !13
  br label %invoke.cont.i.i670

invoke.cont.i.i670:                               ; preds = %if.else.i.i.i667, %if.then.i.i.i665
  %retval.0.i.i.i668 = phi i32 [ %559, %if.then.i.i.i665 ], [ %560, %if.else.i.i.i667 ]
  %cmp3.i.i669 = icmp slt i32 %retval.0.i.i.i668, 1
  br i1 %cmp3.i.i669, label %if.then4.i.i671, label %_ZNSsD2Ev.exit672

if.then4.i.i671:                                  ; preds = %invoke.cont.i.i670
  call void @_ZNSs4_Rep10_M_destroyERKSaIcE(%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"* nonnull %556, %"class.std::allocator"* nonnull dereferenceable(1) %ref.tmp.i659)
  br label %_ZNSsD2Ev.exit672

_ZNSsD2Ev.exit672:                                ; preds = %if.then4.i.i671, %invoke.cont.i.i670, %ehcleanup401
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %557)
  br label %ehcleanup402

ehcleanup402:                                     ; preds = %_ZNSsD2Ev.exit672, %lpad391
  %ehselector.slot.12 = phi i32 [ %ehselector.slot.11, %_ZNSsD2Ev.exit672 ], [ %533, %lpad391 ]
  %exn.slot.12 = phi i8* [ %exn.slot.11, %_ZNSsD2Ev.exit672 ], [ %532, %lpad391 ]
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %84)
  %561 = load i8*, i8** %_M_p.i.i.i.i, align 8, !tbaa !9
  %arrayidx.i.i641 = getelementptr inbounds i8, i8* %561, i64 -24
  %562 = bitcast i8* %arrayidx.i.i641 to %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"*
  %563 = getelementptr inbounds %"class.std::allocator", %"class.std::allocator"* %ref.tmp.i639, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %563)
  %cmp.i.i642 = icmp eq i8* %arrayidx.i.i641, bitcast ([0 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE to i8*)
  br i1 %cmp.i.i642, label %_ZNSsD2Ev.exit652, label %if.then.i.i644, !prof !12

if.then.i.i644:                                   ; preds = %ehcleanup402
  %_M_refcount.i.i643 = getelementptr inbounds i8, i8* %561, i64 -8
  %564 = bitcast i8* %_M_refcount.i.i643 to i32*
  br i1 icmp ne (i8* bitcast (i32 (i32*, void (i8*)*)* @__pthread_key_create to i8*), i8* null), label %if.then.i.i.i645, label %if.else.i.i.i647

if.then.i.i.i645:                                 ; preds = %if.then.i.i644
  %565 = atomicrmw volatile add i32* %564, i32 -1 acq_rel
  br label %invoke.cont.i.i650

if.else.i.i.i647:                                 ; preds = %if.then.i.i644
  %566 = load i32, i32* %564, align 4, !tbaa !13
  %add.i.i.i.i646 = add nsw i32 %566, -1
  store i32 %add.i.i.i.i646, i32* %564, align 4, !tbaa !13
  br label %invoke.cont.i.i650

invoke.cont.i.i650:                               ; preds = %if.else.i.i.i647, %if.then.i.i.i645
  %retval.0.i.i.i648 = phi i32 [ %565, %if.then.i.i.i645 ], [ %566, %if.else.i.i.i647 ]
  %cmp3.i.i649 = icmp slt i32 %retval.0.i.i.i648, 1
  br i1 %cmp3.i.i649, label %if.then4.i.i651, label %_ZNSsD2Ev.exit652

if.then4.i.i651:                                  ; preds = %invoke.cont.i.i650
  call void @_ZNSs4_Rep10_M_destroyERKSaIcE(%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep"* nonnull %562, %"class.std::allocator"* nonnull dereferenceable(1) %ref.tmp.i639)
  br label %_ZNSsD2Ev.exit652

_ZNSsD2Ev.exit652:                                ; preds = %if.then4.i.i651, %invoke.cont.i.i650, %ehcleanup402
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %563)
  br label %ehcleanup404

ehcleanup404:                                     ; preds = %_ZNSsD2Ev.exit652, %lpad387.body
  %ehselector.slot.13 = phi i32 [ %ehselector.slot.12, %_ZNSsD2Ev.exit652 ], [ %530, %lpad387.body ]
  %exn.slot.13 = phi i8* [ %exn.slot.12, %_ZNSsD2Ev.exit652 ], [ %529, %lpad387.body ]
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %83)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %82)
  %.pre1889 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %last_img, i64 0, i32 9
  br label %ehcleanup411

cleanup.cont:                                     ; preds = %_ZN2cv3MatD2Ev.exit941, %_ZNSsD2Ev.exit
  ret void

ehcleanup411:                                     ; preds = %ehcleanup404, %ehcleanup363
  %u.i.i604.pre-phi = phi %"struct.cv::UMatData"** [ %.pre1889, %ehcleanup404 ], [ %u.i.i1117, %ehcleanup363 ]
  %ehselector.slot.15 = phi i32 [ %ehselector.slot.13, %ehcleanup404 ], [ %ehselector.slot.7, %ehcleanup363 ]
  %exn.slot.15 = phi i8* [ %exn.slot.13, %ehcleanup404 ], [ %exn.slot.7, %ehcleanup363 ]
  %567 = load %"struct.cv::UMatData"*, %"struct.cv::UMatData"** %u.i.i604.pre-phi, align 8, !tbaa !76
  %tobool.i.i605 = icmp eq %"struct.cv::UMatData"* %567, null
  br i1 %tobool.i.i605, label %if.end.i.i613, label %land.lhs.true.i.i608

land.lhs.true.i.i608:                             ; preds = %ehcleanup411
  %refcount.i.i606 = getelementptr inbounds %"struct.cv::UMatData", %"struct.cv::UMatData"* %567, i64 0, i32 3
  %568 = atomicrmw add i32* %refcount.i.i606, i32 -1 acq_rel
  %cmp.i.i607 = icmp eq i32 %568, 1
  br i1 %cmp.i.i607, label %if.then.i.i609, label %if.end.i.i613

if.then.i.i609:                                   ; preds = %land.lhs.true.i.i608
  invoke void @_ZN2cv3Mat10deallocateEv(%"class.cv::Mat"* nonnull %last_img)
          to label %if.end.i.i613 unwind label %terminate.lpad.i626

if.end.i.i613:                                    ; preds = %if.then.i.i609, %land.lhs.true.i.i608, %ehcleanup411
  store %"struct.cv::UMatData"* null, %"struct.cv::UMatData"** %u.i.i604.pre-phi, align 8, !tbaa !76
  %data.i.i610 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %last_img, i64 0, i32 4
  %569 = bitcast i8** %data.i.i610 to i8*
  call void @llvm.memset.p0i8.i64(i8* nonnull %569, i8 0, i64 32, i32 8, i1 false)
  %570 = load i32, i32* %dims.i, align 4, !tbaa !24
  %cmp48.i.i612 = icmp sgt i32 %570, 0
  br i1 %cmp48.i.i612, label %for.body.lr.ph.i.i615, label %invoke.cont.i624

for.body.lr.ph.i.i615:                            ; preds = %if.end.i.i613
  %571 = load i32*, i32** %p.i.i603, align 8, !tbaa !77
  br label %for.body.i.i620

for.body.i.i620:                                  ; preds = %for.body.i.i620, %for.body.lr.ph.i.i615
  %indvars.iv.i.i616 = phi i64 [ 0, %for.body.lr.ph.i.i615 ], [ %indvars.iv.next.i.i618, %for.body.i.i620 ]
  %arrayidx.i.i617 = getelementptr inbounds i32, i32* %571, i64 %indvars.iv.i.i616
  store i32 0, i32* %arrayidx.i.i617, align 4, !tbaa !13
  %indvars.iv.next.i.i618 = add nuw nsw i64 %indvars.iv.i.i616, 1
  %572 = load i32, i32* %dims.i, align 4, !tbaa !24
  %573 = sext i32 %572 to i64
  %cmp4.i.i619 = icmp slt i64 %indvars.iv.next.i.i618, %573
  br i1 %cmp4.i.i619, label %for.body.i.i620, label %invoke.cont.i624

invoke.cont.i624:                                 ; preds = %for.body.i.i620, %if.end.i.i613
  %574 = load i64*, i64** %p.i3.i, align 8, !tbaa !34
  %cmp.i623 = icmp eq i64* %574, %arraydecay.i.i
  br i1 %cmp.i623, label %ehcleanup412, label %if.then.i625

if.then.i625:                                     ; preds = %invoke.cont.i624
  %575 = bitcast i64* %574 to i8*
  invoke void @_ZN2cv8fastFreeEPv(i8* %575)
          to label %ehcleanup412 unwind label %terminate.lpad.i626

terminate.lpad.i626:                              ; preds = %if.then.i625, %if.then.i.i609
  %576 = landingpad { i8*, i32 }
          catch i8* null
  %577 = extractvalue { i8*, i32 } %576, 0
  call void @__clang_call_terminate(i8* %577)
  unreachable

ehcleanup412:                                     ; preds = %if.then.i625, %invoke.cont.i624
  call void @llvm.lifetime.end.p0i8(i64 96, i8* nonnull %31)
  %u.i.i572 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %img, i64 0, i32 9
  %578 = load %"struct.cv::UMatData"*, %"struct.cv::UMatData"** %u.i.i572, align 8, !tbaa !76
  %tobool.i.i573 = icmp eq %"struct.cv::UMatData"* %578, null
  br i1 %tobool.i.i573, label %if.end.i.i581, label %land.lhs.true.i.i576

land.lhs.true.i.i576:                             ; preds = %ehcleanup412
  %refcount.i.i574 = getelementptr inbounds %"struct.cv::UMatData", %"struct.cv::UMatData"* %578, i64 0, i32 3
  %579 = atomicrmw add i32* %refcount.i.i574, i32 -1 acq_rel
  %cmp.i.i575 = icmp eq i32 %579, 1
  br i1 %cmp.i.i575, label %if.then.i.i577, label %if.end.i.i581

if.then.i.i577:                                   ; preds = %land.lhs.true.i.i576
  invoke void @_ZN2cv3Mat10deallocateEv(%"class.cv::Mat"* nonnull %img)
          to label %if.end.i.i581 unwind label %terminate.lpad.i594

if.end.i.i581:                                    ; preds = %if.then.i.i577, %land.lhs.true.i.i576, %ehcleanup412
  store %"struct.cv::UMatData"* null, %"struct.cv::UMatData"** %u.i.i572, align 8, !tbaa !76
  %data.i.i578 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %img, i64 0, i32 4
  %dims.i.i579 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %img, i64 0, i32 1
  %580 = bitcast i8** %data.i.i578 to i8*
  call void @llvm.memset.p0i8.i64(i8* nonnull %580, i8 0, i64 32, i32 8, i1 false)
  %581 = load i32, i32* %dims.i.i579, align 4, !tbaa !24
  %cmp48.i.i580 = icmp sgt i32 %581, 0
  br i1 %cmp48.i.i580, label %for.body.lr.ph.i.i583, label %invoke.cont.i592

for.body.lr.ph.i.i583:                            ; preds = %if.end.i.i581
  %p.i.i582 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %img, i64 0, i32 10, i32 0
  %582 = load i32*, i32** %p.i.i582, align 8, !tbaa !77
  br label %for.body.i.i588

for.body.i.i588:                                  ; preds = %for.body.i.i588, %for.body.lr.ph.i.i583
  %indvars.iv.i.i584 = phi i64 [ 0, %for.body.lr.ph.i.i583 ], [ %indvars.iv.next.i.i586, %for.body.i.i588 ]
  %arrayidx.i.i585 = getelementptr inbounds i32, i32* %582, i64 %indvars.iv.i.i584
  store i32 0, i32* %arrayidx.i.i585, align 4, !tbaa !13
  %indvars.iv.next.i.i586 = add nuw nsw i64 %indvars.iv.i.i584, 1
  %583 = load i32, i32* %dims.i.i579, align 4, !tbaa !24
  %584 = sext i32 %583 to i64
  %cmp4.i.i587 = icmp slt i64 %indvars.iv.next.i.i586, %584
  br i1 %cmp4.i.i587, label %for.body.i.i588, label %invoke.cont.i592

invoke.cont.i592:                                 ; preds = %for.body.i.i588, %if.end.i.i581
  %p.i589 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %img, i64 0, i32 11, i32 0
  %585 = load i64*, i64** %p.i589, align 8, !tbaa !34
  %arraydecay.i590 = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %img, i64 0, i32 11, i32 1, i64 0
  %cmp.i591 = icmp eq i64* %585, %arraydecay.i590
  br i1 %cmp.i591, label %ehcleanup414, label %if.then.i593

if.then.i593:                                     ; preds = %invoke.cont.i592
  %586 = bitcast i64* %585 to i8*
  invoke void @_ZN2cv8fastFreeEPv(i8* %586)
          to label %ehcleanup414 unwind label %terminate.lpad.i594

terminate.lpad.i594:                              ; preds = %if.then.i593, %if.then.i.i577
  %587 = landingpad { i8*, i32 }
          catch i8* null
  %588 = extractvalue { i8*, i32 } %587, 0
  call void @__clang_call_terminate(i8* %588)
  unreachable

ehcleanup414:                                     ; preds = %if.then.i593, %invoke.cont.i592, %lpad10
  %ehselector.slot.17 = phi i32 [ %94, %lpad10 ], [ %ehselector.slot.15, %invoke.cont.i592 ], [ %ehselector.slot.15, %if.then.i593 ]
  %exn.slot.17 = phi i8* [ %93, %lpad10 ], [ %exn.slot.15, %invoke.cont.i592 ], [ %exn.slot.15, %if.then.i593 ]
  call void @llvm.lifetime.end.p0i8(i64 96, i8* nonnull %26)
  %u.i.i = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %next_section_img, i64 0, i32 9
  %589 = load %"struct.cv::UMatData"*, %"struct.cv::UMatData"** %u.i.i, align 8, !tbaa !76
  %tobool.i.i = icmp eq %"struct.cv::UMatData"* %589, null
  br i1 %tobool.i.i, label %if.end.i.i, label %land.lhs.true.i.i

land.lhs.true.i.i:                                ; preds = %ehcleanup414
  %refcount.i.i = getelementptr inbounds %"struct.cv::UMatData", %"struct.cv::UMatData"* %589, i64 0, i32 3
  %590 = atomicrmw add i32* %refcount.i.i, i32 -1 acq_rel
  %cmp.i.i = icmp eq i32 %590, 1
  br i1 %cmp.i.i, label %if.then.i.i, label %if.end.i.i

if.then.i.i:                                      ; preds = %land.lhs.true.i.i
  invoke void @_ZN2cv3Mat10deallocateEv(%"class.cv::Mat"* nonnull %next_section_img)
          to label %if.end.i.i unwind label %terminate.lpad.i

if.end.i.i:                                       ; preds = %if.then.i.i, %land.lhs.true.i.i, %ehcleanup414
  store %"struct.cv::UMatData"* null, %"struct.cv::UMatData"** %u.i.i, align 8, !tbaa !76
  %data.i.i = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %next_section_img, i64 0, i32 4
  %dims.i.i = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %next_section_img, i64 0, i32 1
  %591 = bitcast i8** %data.i.i to i8*
  call void @llvm.memset.p0i8.i64(i8* nonnull %591, i8 0, i64 32, i32 16, i1 false)
  %592 = load i32, i32* %dims.i.i, align 4, !tbaa !24
  %cmp48.i.i = icmp sgt i32 %592, 0
  br i1 %cmp48.i.i, label %for.body.lr.ph.i.i, label %invoke.cont.i

for.body.lr.ph.i.i:                               ; preds = %if.end.i.i
  %p.i.i = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %next_section_img, i64 0, i32 10, i32 0
  %593 = load i32*, i32** %p.i.i, align 16, !tbaa !77
  br label %for.body.i.i

for.body.i.i:                                     ; preds = %for.body.i.i, %for.body.lr.ph.i.i
  %indvars.iv.i.i = phi i64 [ 0, %for.body.lr.ph.i.i ], [ %indvars.iv.next.i.i, %for.body.i.i ]
  %arrayidx.i.i = getelementptr inbounds i32, i32* %593, i64 %indvars.iv.i.i
  store i32 0, i32* %arrayidx.i.i, align 4, !tbaa !13
  %indvars.iv.next.i.i = add nuw nsw i64 %indvars.iv.i.i, 1
  %594 = load i32, i32* %dims.i.i, align 4, !tbaa !24
  %595 = sext i32 %594 to i64
  %cmp4.i.i = icmp slt i64 %indvars.iv.next.i.i, %595
  br i1 %cmp4.i.i, label %for.body.i.i, label %invoke.cont.i

invoke.cont.i:                                    ; preds = %for.body.i.i, %if.end.i.i
  %p.i = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %next_section_img, i64 0, i32 11, i32 0
  %596 = load i64*, i64** %p.i, align 8, !tbaa !34
  %arraydecay.i = getelementptr inbounds %"class.cv::Mat", %"class.cv::Mat"* %next_section_img, i64 0, i32 11, i32 1, i64 0
  %cmp.i = icmp eq i64* %596, %arraydecay.i
  br i1 %cmp.i, label %_ZN2cv3MatD2Ev.exit, label %if.then.i

if.then.i:                                        ; preds = %invoke.cont.i
  %597 = bitcast i64* %596 to i8*
  invoke void @_ZN2cv8fastFreeEPv(i8* %597)
          to label %_ZN2cv3MatD2Ev.exit unwind label %terminate.lpad.i

terminate.lpad.i:                                 ; preds = %if.then.i, %if.then.i.i
  %598 = landingpad { i8*, i32 }
          catch i8* null
  %599 = extractvalue { i8*, i32 } %598, 0
  call void @__clang_call_terminate(i8* %599)
  unreachable

_ZN2cv3MatD2Ev.exit:                              ; preds = %if.then.i, %invoke.cont.i
  call void @llvm.lifetime.end.p0i8(i64 96, i8* nonnull %23)
  br label %eh.resume

eh.resume:                                        ; preds = %_ZN2cv3MatD2Ev.exit, %_ZNSsD2Ev.exit565
  %ehselector.slot.18 = phi i32 [ %14, %_ZNSsD2Ev.exit565 ], [ %ehselector.slot.17, %_ZN2cv3MatD2Ev.exit ]
  %exn.slot.18 = phi i8* [ %13, %_ZNSsD2Ev.exit565 ], [ %exn.slot.17, %_ZN2cv3MatD2Ev.exit ]
  %lpad.val420 = insertvalue { i8*, i32 } undef, i8* %exn.slot.18, 0
  %lpad.val421 = insertvalue { i8*, i32 } %lpad.val420, i32 %ehselector.slot.18, 1
  resume { i8*, i32 } %lpad.val421

if.then101.us.1:                                  ; preds = %for.inc163.us
  %conv106.us.1 = zext i8 %167 to i16
  %600 = load i8*, i8** %data.i697, align 8, !tbaa !33
  %601 = load i64*, i64** %p.i3.i638, align 8, !tbaa !34
  %602 = load i64, i64* %601, align 8, !tbaa !25
  %mul.i801.us.1 = mul i64 %602, %indvars.iv1867
  %add.ptr.i802.us.1 = getelementptr inbounds i8, i8* %600, i64 %mul.i801.us.1
  %603 = bitcast i8* %add.ptr.i802.us.1 to i16*
  %arrayidx2.i804.us.1 = getelementptr inbounds i16, i16* %603, i64 %129
  %604 = load i16, i16* %arrayidx2.i804.us.1, align 2, !tbaa !29
  %add110.us.1 = add i16 %604, %conv106.us.1
  store i16 %add110.us.1, i16* %arrayidx2.i804.us.1, align 2, !tbaa !29
  %605 = load i8*, i8** %data.i714, align 8, !tbaa !33
  %606 = load i64*, i64** %p.i3.i658, align 8, !tbaa !34
  %607 = load i64, i64* %606, align 8, !tbaa !25
  %mul.i822.us.1 = mul i64 %607, %indvars.iv1867
  %add.ptr.i823.us.1 = getelementptr inbounds i8, i8* %605, i64 %mul.i822.us.1
  %608 = bitcast i8* %add.ptr.i823.us.1 to i16*
  %arrayidx2.i825.us.1 = getelementptr inbounds i16, i16* %608, i64 %129
  %609 = load i16, i16* %arrayidx2.i825.us.1, align 2, !tbaa !29
  %add115.us.1 = add i16 %609, 1
  store i16 %add115.us.1, i16* %arrayidx2.i825.us.1, align 2, !tbaa !29
  br label %if.end141.us.1

if.end141.us.1:                                   ; preds = %if.then101.us.1, %for.inc163.us
  %610 = load i8, i8* %arrayidx2.i741, align 1, !tbaa !35
  %cmp145.us.1 = icmp eq i8 %610, 0
  br i1 %cmp145.us.1, label %for.inc163.us.1, label %if.then146.us.1

if.then146.us.1:                                  ; preds = %if.end141.us.1
  %arrayidx2.i875.us.1 = getelementptr inbounds i8, i8* %add.ptr.i873.us.1, i64 %140
  %611 = load i8, i8* %arrayidx2.i875.us.1, align 1, !tbaa !35
  %conv151.us.1 = zext i8 %611 to i16
  %612 = load i8*, i8** %data.i697, align 8, !tbaa !33
  %613 = load i64*, i64** %p.i3.i638, align 8, !tbaa !34
  %614 = load i64, i64* %613, align 8, !tbaa !25
  %mul.i884.us.1 = mul i64 %614, %indvars.iv1867
  %add.ptr.i885.us.1 = getelementptr inbounds i8, i8* %612, i64 %mul.i884.us.1
  %615 = bitcast i8* %add.ptr.i885.us.1 to i16*
  %arrayidx2.i887.us.1 = getelementptr inbounds i16, i16* %615, i64 %129
  %616 = load i16, i16* %arrayidx2.i887.us.1, align 2, !tbaa !29
  %add155.us.1 = add i16 %616, %conv151.us.1
  store i16 %add155.us.1, i16* %arrayidx2.i887.us.1, align 2, !tbaa !29
  %617 = load i8*, i8** %data.i714, align 8, !tbaa !33
  %618 = load i64*, i64** %p.i3.i658, align 8, !tbaa !34
  %619 = load i64, i64* %618, align 8, !tbaa !25
  %mul.i891.us.1 = mul i64 %619, %indvars.iv1867
  %add.ptr.i892.us.1 = getelementptr inbounds i8, i8* %617, i64 %mul.i891.us.1
  %620 = bitcast i8* %add.ptr.i892.us.1 to i16*
  %arrayidx2.i894.us.1 = getelementptr inbounds i16, i16* %620, i64 %129
  %621 = load i16, i16* %arrayidx2.i894.us.1, align 2, !tbaa !29
  %add160.us.1 = add i16 %621, 1
  store i16 %add160.us.1, i16* %arrayidx2.i894.us.1, align 2, !tbaa !29
  br label %for.inc163.us.1

for.inc163.us.1:                                  ; preds = %if.then146.us.1, %if.end141.us.1
  %mul.i759.us.2 = mul i64 %143, %indvars.iv1867
  %add.ptr.i760.us.2 = getelementptr inbounds i8, i8* %141, i64 %mul.i759.us.2
  %arrayidx2.i762.us.2 = getelementptr inbounds i8, i8* %add.ptr.i760.us.2, i64 %140
  %622 = load i8, i8* %arrayidx2.i762.us.2, align 1, !tbaa !35
  %cmp100.us.2 = icmp eq i8 %622, 0
  br i1 %cmp100.us.2, label %if.end141.us.2, label %if.then101.us.2

if.then101.us.2:                                  ; preds = %for.inc163.us.1
  %conv106.us.2 = zext i8 %622 to i16
  %623 = load i8*, i8** %data.i697, align 8, !tbaa !33
  %624 = load i64*, i64** %p.i3.i638, align 8, !tbaa !34
  %625 = load i64, i64* %624, align 8, !tbaa !25
  %mul.i801.us.2 = mul i64 %625, %indvars.iv1867
  %add.ptr.i802.us.2 = getelementptr inbounds i8, i8* %623, i64 %mul.i801.us.2
  %626 = bitcast i8* %add.ptr.i802.us.2 to i16*
  %arrayidx2.i804.us.2 = getelementptr inbounds i16, i16* %626, i64 %129
  %627 = load i16, i16* %arrayidx2.i804.us.2, align 2, !tbaa !29
  %add110.us.2 = add i16 %627, %conv106.us.2
  store i16 %add110.us.2, i16* %arrayidx2.i804.us.2, align 2, !tbaa !29
  %628 = load i8*, i8** %data.i714, align 8, !tbaa !33
  %629 = load i64*, i64** %p.i3.i658, align 8, !tbaa !34
  %630 = load i64, i64* %629, align 8, !tbaa !25
  %mul.i822.us.2 = mul i64 %630, %indvars.iv1867
  %add.ptr.i823.us.2 = getelementptr inbounds i8, i8* %628, i64 %mul.i822.us.2
  %631 = bitcast i8* %add.ptr.i823.us.2 to i16*
  %arrayidx2.i825.us.2 = getelementptr inbounds i16, i16* %631, i64 %129
  %632 = load i16, i16* %arrayidx2.i825.us.2, align 2, !tbaa !29
  %add115.us.2 = add i16 %632, 1
  store i16 %add115.us.2, i16* %arrayidx2.i825.us.2, align 2, !tbaa !29
  br label %if.end141.us.2

if.end141.us.2:                                   ; preds = %if.then101.us.2, %for.inc163.us.1
  %633 = load i8, i8* %arrayidx2.i741, align 1, !tbaa !35
  %cmp145.us.2 = icmp eq i8 %633, 0
  br i1 %cmp145.us.2, label %for.inc163.us.2, label %if.then146.us.2

if.then146.us.2:                                  ; preds = %if.end141.us.2
  %arrayidx2.i875.us.2 = getelementptr inbounds i8, i8* %add.ptr.i739, i64 %140
  %634 = load i8, i8* %arrayidx2.i875.us.2, align 1, !tbaa !35
  %conv151.us.2 = zext i8 %634 to i16
  %635 = load i8*, i8** %data.i697, align 8, !tbaa !33
  %636 = load i64*, i64** %p.i3.i638, align 8, !tbaa !34
  %637 = load i64, i64* %636, align 8, !tbaa !25
  %mul.i884.us.2 = mul i64 %637, %indvars.iv1867
  %add.ptr.i885.us.2 = getelementptr inbounds i8, i8* %635, i64 %mul.i884.us.2
  %638 = bitcast i8* %add.ptr.i885.us.2 to i16*
  %arrayidx2.i887.us.2 = getelementptr inbounds i16, i16* %638, i64 %129
  %639 = load i16, i16* %arrayidx2.i887.us.2, align 2, !tbaa !29
  %add155.us.2 = add i16 %639, %conv151.us.2
  store i16 %add155.us.2, i16* %arrayidx2.i887.us.2, align 2, !tbaa !29
  %640 = load i8*, i8** %data.i714, align 8, !tbaa !33
  %641 = load i64*, i64** %p.i3.i658, align 8, !tbaa !34
  %642 = load i64, i64* %641, align 8, !tbaa !25
  %mul.i891.us.2 = mul i64 %642, %indvars.iv1867
  %add.ptr.i892.us.2 = getelementptr inbounds i8, i8* %640, i64 %mul.i891.us.2
  %643 = bitcast i8* %add.ptr.i892.us.2 to i16*
  %arrayidx2.i894.us.2 = getelementptr inbounds i16, i16* %643, i64 %129
  %644 = load i16, i16* %arrayidx2.i894.us.2, align 2, !tbaa !29
  %add160.us.2 = add i16 %644, 1
  store i16 %add160.us.2, i16* %arrayidx2.i894.us.2, align 2, !tbaa !29
  br label %for.inc163.us.2

for.inc163.us.2:                                  ; preds = %if.then146.us.2, %if.end141.us.2
  %mul.i759.us.3 = mul i64 %143, %138
  %add.ptr.i760.us.3 = getelementptr inbounds i8, i8* %141, i64 %mul.i759.us.3
  %arrayidx2.i762.us.3 = getelementptr inbounds i8, i8* %add.ptr.i760.us.3, i64 %140
  %645 = load i8, i8* %arrayidx2.i762.us.3, align 1, !tbaa !35
  %cmp100.us.3 = icmp eq i8 %645, 0
  br i1 %cmp100.us.3, label %if.end141.us.3, label %if.then101.us.3

if.then101.us.3:                                  ; preds = %for.inc163.us.2
  %conv106.us.3 = zext i8 %645 to i16
  %646 = load i8*, i8** %data.i697, align 8, !tbaa !33
  %647 = load i64*, i64** %p.i3.i638, align 8, !tbaa !34
  %648 = load i64, i64* %647, align 8, !tbaa !25
  %mul.i801.us.3 = mul i64 %648, %indvars.iv1867
  %add.ptr.i802.us.3 = getelementptr inbounds i8, i8* %646, i64 %mul.i801.us.3
  %649 = bitcast i8* %add.ptr.i802.us.3 to i16*
  %arrayidx2.i804.us.3 = getelementptr inbounds i16, i16* %649, i64 %129
  %650 = load i16, i16* %arrayidx2.i804.us.3, align 2, !tbaa !29
  %add110.us.3 = add i16 %650, %conv106.us.3
  store i16 %add110.us.3, i16* %arrayidx2.i804.us.3, align 2, !tbaa !29
  %651 = load i8*, i8** %data.i714, align 8, !tbaa !33
  %652 = load i64*, i64** %p.i3.i658, align 8, !tbaa !34
  %653 = load i64, i64* %652, align 8, !tbaa !25
  %mul.i822.us.3 = mul i64 %653, %indvars.iv1867
  %add.ptr.i823.us.3 = getelementptr inbounds i8, i8* %651, i64 %mul.i822.us.3
  %654 = bitcast i8* %add.ptr.i823.us.3 to i16*
  %arrayidx2.i825.us.3 = getelementptr inbounds i16, i16* %654, i64 %129
  %655 = load i16, i16* %arrayidx2.i825.us.3, align 2, !tbaa !29
  %add115.us.3 = add i16 %655, 1
  store i16 %add115.us.3, i16* %arrayidx2.i825.us.3, align 2, !tbaa !29
  br label %if.end141.us.3

if.end141.us.3:                                   ; preds = %if.then101.us.3, %for.inc163.us.2
  %656 = load i8, i8* %arrayidx2.i741, align 1, !tbaa !35
  %cmp145.us.3 = icmp eq i8 %656, 0
  br i1 %cmp145.us.3, label %for.inc163.us.3, label %if.then146.us.3

if.then146.us.3:                                  ; preds = %if.end141.us.3
  %arrayidx2.i875.us.3 = getelementptr inbounds i8, i8* %add.ptr.i873.us.3, i64 %140
  %657 = load i8, i8* %arrayidx2.i875.us.3, align 1, !tbaa !35
  %conv151.us.3 = zext i8 %657 to i16
  %658 = load i8*, i8** %data.i697, align 8, !tbaa !33
  %659 = load i64*, i64** %p.i3.i638, align 8, !tbaa !34
  %660 = load i64, i64* %659, align 8, !tbaa !25
  %mul.i884.us.3 = mul i64 %660, %indvars.iv1867
  %add.ptr.i885.us.3 = getelementptr inbounds i8, i8* %658, i64 %mul.i884.us.3
  %661 = bitcast i8* %add.ptr.i885.us.3 to i16*
  %arrayidx2.i887.us.3 = getelementptr inbounds i16, i16* %661, i64 %129
  %662 = load i16, i16* %arrayidx2.i887.us.3, align 2, !tbaa !29
  %add155.us.3 = add i16 %662, %conv151.us.3
  store i16 %add155.us.3, i16* %arrayidx2.i887.us.3, align 2, !tbaa !29
  %663 = load i8*, i8** %data.i714, align 8, !tbaa !33
  %664 = load i64*, i64** %p.i3.i658, align 8, !tbaa !34
  %665 = load i64, i64* %664, align 8, !tbaa !25
  %mul.i891.us.3 = mul i64 %665, %indvars.iv1867
  %add.ptr.i892.us.3 = getelementptr inbounds i8, i8* %663, i64 %mul.i891.us.3
  %666 = bitcast i8* %add.ptr.i892.us.3 to i16*
  %arrayidx2.i894.us.3 = getelementptr inbounds i16, i16* %666, i64 %129
  %667 = load i16, i16* %arrayidx2.i894.us.3, align 2, !tbaa !29
  %add160.us.3 = add i16 %667, 1
  store i16 %add160.us.3, i16* %arrayidx2.i894.us.3, align 2, !tbaa !29
  br label %for.inc163.us.3

for.inc163.us.3:                                  ; preds = %if.then146.us.3, %if.end141.us.3
  %mul.i759.us.4 = mul i64 %143, %139
  %add.ptr.i760.us.4 = getelementptr inbounds i8, i8* %141, i64 %mul.i759.us.4
  %arrayidx2.i762.us.4 = getelementptr inbounds i8, i8* %add.ptr.i760.us.4, i64 %140
  %668 = load i8, i8* %arrayidx2.i762.us.4, align 1, !tbaa !35
  %cmp100.us.4 = icmp eq i8 %668, 0
  br i1 %cmp100.us.4, label %if.end141.us.4, label %if.then101.us.4

if.then101.us.4:                                  ; preds = %for.inc163.us.3
  %conv106.us.4 = zext i8 %668 to i16
  %669 = load i8*, i8** %data.i697, align 8, !tbaa !33
  %670 = load i64*, i64** %p.i3.i638, align 8, !tbaa !34
  %671 = load i64, i64* %670, align 8, !tbaa !25
  %mul.i801.us.4 = mul i64 %671, %indvars.iv1867
  %add.ptr.i802.us.4 = getelementptr inbounds i8, i8* %669, i64 %mul.i801.us.4
  %672 = bitcast i8* %add.ptr.i802.us.4 to i16*
  %arrayidx2.i804.us.4 = getelementptr inbounds i16, i16* %672, i64 %129
  %673 = load i16, i16* %arrayidx2.i804.us.4, align 2, !tbaa !29
  %add110.us.4 = add i16 %673, %conv106.us.4
  store i16 %add110.us.4, i16* %arrayidx2.i804.us.4, align 2, !tbaa !29
  %674 = load i8*, i8** %data.i714, align 8, !tbaa !33
  %675 = load i64*, i64** %p.i3.i658, align 8, !tbaa !34
  %676 = load i64, i64* %675, align 8, !tbaa !25
  %mul.i822.us.4 = mul i64 %676, %indvars.iv1867
  %add.ptr.i823.us.4 = getelementptr inbounds i8, i8* %674, i64 %mul.i822.us.4
  %677 = bitcast i8* %add.ptr.i823.us.4 to i16*
  %arrayidx2.i825.us.4 = getelementptr inbounds i16, i16* %677, i64 %129
  %678 = load i16, i16* %arrayidx2.i825.us.4, align 2, !tbaa !29
  %add115.us.4 = add i16 %678, 1
  store i16 %add115.us.4, i16* %arrayidx2.i825.us.4, align 2, !tbaa !29
  br label %if.end141.us.4

if.end141.us.4:                                   ; preds = %if.then101.us.4, %for.inc163.us.3
  %679 = load i8, i8* %arrayidx2.i741, align 1, !tbaa !35
  %cmp145.us.4 = icmp eq i8 %679, 0
  br i1 %cmp145.us.4, label %for.cond.cleanup92, label %if.then146.us.4

if.then146.us.4:                                  ; preds = %if.end141.us.4
  %arrayidx2.i875.us.4 = getelementptr inbounds i8, i8* %add.ptr.i873.us.4, i64 %140
  %680 = load i8, i8* %arrayidx2.i875.us.4, align 1, !tbaa !35
  %conv151.us.4 = zext i8 %680 to i16
  %681 = load i8*, i8** %data.i697, align 8, !tbaa !33
  %682 = load i64*, i64** %p.i3.i638, align 8, !tbaa !34
  %683 = load i64, i64* %682, align 8, !tbaa !25
  %mul.i884.us.4 = mul i64 %683, %indvars.iv1867
  %add.ptr.i885.us.4 = getelementptr inbounds i8, i8* %681, i64 %mul.i884.us.4
  %684 = bitcast i8* %add.ptr.i885.us.4 to i16*
  %arrayidx2.i887.us.4 = getelementptr inbounds i16, i16* %684, i64 %129
  %685 = load i16, i16* %arrayidx2.i887.us.4, align 2, !tbaa !29
  %add155.us.4 = add i16 %685, %conv151.us.4
  store i16 %add155.us.4, i16* %arrayidx2.i887.us.4, align 2, !tbaa !29
  %686 = load i8*, i8** %data.i714, align 8, !tbaa !33
  %687 = load i64*, i64** %p.i3.i658, align 8, !tbaa !34
  %688 = load i64, i64* %687, align 8, !tbaa !25
  %mul.i891.us.4 = mul i64 %688, %indvars.iv1867
  %add.ptr.i892.us.4 = getelementptr inbounds i8, i8* %686, i64 %mul.i891.us.4
  %689 = bitcast i8* %add.ptr.i892.us.4 to i16*
  %arrayidx2.i894.us.4 = getelementptr inbounds i16, i16* %689, i64 %129
  %690 = load i16, i16* %arrayidx2.i894.us.4, align 2, !tbaa !29
  %add160.us.4 = add i16 %690, 1
  store i16 %add160.us.4, i16* %arrayidx2.i894.us.4, align 2, !tbaa !29
  br label %for.cond.cleanup92
}

; CHECK: define void @_ZN3tfk6Render23render_stack_with_patchEPNS_5StackESt4pairIN2cv6Point_IfEES6_ENS_10ResolutionESs(
; CHECK: csi.cleanup:

; CHECK-LABEL: define private fastcc void @_ZN3tfk6Render23render_stack_with_patchEPNS_5StackESt4pairIN2cv6Point_IfEES6_ENS_10ResolutionESs.outline_pfor.detach236.ls3(
; CHECK: unnamed_addr #[[ATTRIBUTE:[0-9]+]]
; CHECK: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[CONTINUE:.+]]
; CHECK: [[DETACHED]]:
; CHECK: call void @__csi_task(
; CHECK: call fastcc void @_ZN3tfk6Render23render_stack_with_patchEPNS_5StackESt4pairIN2cv6Point_IfEES6_ENS_10ResolutionESs.outline_pfor.detach236.ls3(
; CHECK: call void @__csi_task_exit(
; CHECK: reattach within %[[SYNCREG]], label %[[CONTINUE]]

; CHECK-LABEL: define private fastcc void @_ZN3tfk6Render23render_stack_with_patchEPNS_5StackESt4pairIN2cv6Point_IfEES6_ENS_10ResolutionESs.outline_pfor.detach216.ls2(
; CHECK: unnamed_addr #[[ATTRIBUTE]]
; CHECK: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[CONTINUE:.+]]
; CHECK: [[DETACHED]]:
; CHECK: call void @__csi_task(
; CHECK: call fastcc void @_ZN3tfk6Render23render_stack_with_patchEPNS_5StackESt4pairIN2cv6Point_IfEES6_ENS_10ResolutionESs.outline_pfor.detach216.ls2(
; CHECK: call void @__csi_task_exit(
; CHECK: reattach within %[[SYNCREG]], label %[[CONTINUE]]

; CHECK-LABEL: define private fastcc void @_ZN3tfk6Render23render_stack_with_patchEPNS_5StackESt4pairIN2cv6Point_IfEES6_ENS_10ResolutionESs.outline_pfor.detach64.ls2(
; CHECK: unnamed_addr #[[ATTRIBUTE]]
; CHECK: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[CONTINUE:.+]]
; CHECK: [[DETACHED]]:
; CHECK: call void @__csi_task(
; CHECK: call fastcc void @_ZN3tfk6Render23render_stack_with_patchEPNS_5StackESt4pairIN2cv6Point_IfEES6_ENS_10ResolutionESs.outline_pfor.detach64.ls2(
; CHECK: call void @__csi_task_exit(
; CHECK: reattach within %[[SYNCREG]], label %[[CONTINUE]]

; CHECK-LABEL: define private fastcc void @_ZN3tfk6Render23render_stack_with_patchEPNS_5StackESt4pairIN2cv6Point_IfEES6_ENS_10ResolutionESs.outline_pfor.detach.us.ls2(
; CHECK: unnamed_addr #[[ATTRIBUTE]]
; CHECK: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[CONTINUE:.+]]
; CHECK: [[DETACHED]]:
; CHECK: call void @__csi_task(
; CHECK: call fastcc void @_ZN3tfk6Render23render_stack_with_patchEPNS_5StackESt4pairIN2cv6Point_IfEES6_ENS_10ResolutionESs.outline_pfor.detach.us.ls2(
; CHECK: call void @__csi_task_exit(
; CHECK: reattach within %[[SYNCREG]], label %[[CONTINUE]]

attributes #0 = { argmemonly nounwind }
attributes #1 = { nounwind readnone speculatable }

; CHECK: attributes #[[ATTRIBUTE]] = { nounwind }

!0 = !{!1, !3, i64 8}
!1 = !{!"_ZTSSt12_Vector_baseIPN3tfk7SectionESaIS2_EE", !2, i64 0}
!2 = !{!"_ZTSNSt12_Vector_baseIPN3tfk7SectionESaIS2_EE12_Vector_implE", !3, i64 0, !3, i64 8, !3, i64 16}
!3 = !{!"any pointer", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = !{!1, !3, i64 0}
!7 = !{!8, !8, i64 0}
!8 = !{!"float", !4, i64 0}
!9 = !{!10, !3, i64 0}
!10 = !{!"_ZTSSs", !11, i64 0}
!11 = !{!"_ZTSNSs12_Alloc_hiderE", !3, i64 0}
!12 = !{!"branch_weights", i32 2000, i32 1}
!13 = !{!14, !14, i64 0}
!14 = !{!"int", !4, i64 0}
!15 = !{!3, !3, i64 0}
!16 = !{!17, !14, i64 0}
!17 = !{!"_ZTSN2cv3MatE", !14, i64 0, !14, i64 4, !14, i64 8, !14, i64 12, !3, i64 16, !3, i64 24, !3, i64 32, !3, i64 40, !3, i64 48, !3, i64 56, !18, i64 64, !19, i64 72}
!18 = !{!"_ZTSN2cv7MatSizeE", !3, i64 0}
!19 = !{!"_ZTSN2cv7MatStepE", !3, i64 0, !4, i64 8}
!20 = !{!18, !3, i64 0}
!21 = !{!19, !3, i64 0}
!22 = !{!17, !14, i64 8}
!23 = !{!17, !14, i64 12}
!24 = !{!17, !14, i64 4}
!25 = !{!26, !26, i64 0}
!26 = !{!"long", !4, i64 0}
!27 = distinct !{!27, !28}
!28 = !{!"tapir.loop.spawn.strategy", i32 1}
!29 = !{!30, !30, i64 0}
!30 = !{!"short", !4, i64 0}
!31 = distinct !{!31, !32}
!32 = !{!"llvm.loop.unroll.disable"}
!33 = !{!17, !3, i64 16}
!34 = !{!17, !3, i64 72}
!35 = !{!4, !4, i64 0}
!36 = distinct !{!36, !28}
!37 = distinct !{!37, !28}
!38 = distinct !{!38, !28}
!39 = !{!40}
!40 = distinct !{!40, !41, !"_ZStplIcSt11char_traitsIcESaIcEESbIT_T0_T1_ERKS6_PKS3_: %agg.result"}
!41 = distinct !{!41, !"_ZStplIcSt11char_traitsIcESaIcEESbIT_T0_T1_ERKS6_PKS3_"}
!42 = !{!43, !14, i64 4}
!43 = !{!"_ZTSN3tfk7SectionE", !14, i64 0, !14, i64 4, !14, i64 8, !14, i64 12, !14, i64 16, !44, i64 20, !14, i64 24, !14, i64 28, !44, i64 32, !45, i64 36, !45, i64 52, !3, i64 72, !3, i64 80, !10, i64 88, !10, i64 96, !3, i64 104, !47, i64 112, !48, i64 136, !3, i64 160, !3, i64 168, !49, i64 176, !49, i64 184, !49, i64 192, !49, i64 200, !49, i64 208, !49, i64 216, !3, i64 224, !3, i64 232, !3, i64 240, !3, i64 248, !3, i64 256, !3, i64 264, !3, i64 272, !50, i64 280, !56, i64 328, !17, i64 352, !17, i64 448, !17, i64 544, !14, i64 640, !14, i64 644, !3, i64 648, !3, i64 656, !3, i64 664}
!44 = !{!"bool", !4, i64 0}
!45 = !{!"_ZTSSt4pairIN2cv6Point_IfEES2_E", !46, i64 0, !46, i64 8}
!46 = !{!"_ZTSN2cv6Point_IfEE", !8, i64 0, !8, i64 4}
!47 = !{!"_ZTSSt6vectorIN2cv3MatESaIS1_EE"}
!48 = !{!"_ZTSSt6vectorIPN3tfk4TileESaIS2_EE"}
!49 = !{!"double", !4, i64 0}
!50 = !{!"_ZTSSt3setIiSt4lessIiESaIiEE", !51, i64 0}
!51 = !{!"_ZTSSt8_Rb_treeIiiSt9_IdentityIiESt4lessIiESaIiEE", !52, i64 0}
!52 = !{!"_ZTSNSt8_Rb_treeIiiSt9_IdentityIiESt4lessIiESaIiEE13_Rb_tree_implIS3_Lb1EEE", !53, i64 0, !54, i64 8, !26, i64 40}
!53 = !{!"_ZTSSt4lessIiE"}
!54 = !{!"_ZTSSt18_Rb_tree_node_base", !55, i64 0, !3, i64 8, !3, i64 16, !3, i64 24}
!55 = !{!"_ZTSSt14_Rb_tree_color", !4, i64 0}
!56 = !{!"_ZTSSt6vectorI8tfkMatchSaIS0_EE"}
!57 = !{!58}
!58 = distinct !{!58, !59, !"_ZStplIcSt11char_traitsIcESaIcEESbIT_T0_T1_EOS6_S7_: %agg.result"}
!59 = distinct !{!59, !"_ZStplIcSt11char_traitsIcESaIcEESbIT_T0_T1_EOS6_S7_"}
!60 = !{!61, !26, i64 0}
!61 = !{!"_ZTSNSs9_Rep_baseE", !26, i64 0, !26, i64 8, !14, i64 16}
!62 = !{!61, !26, i64 8}
!63 = !{!11, !3, i64 0}
!64 = !{!65}
!65 = distinct !{!65, !66, !"_ZStplIcSt11char_traitsIcESaIcEESbIT_T0_T1_EOS6_PKS3_: %agg.result"}
!66 = distinct !{!66, !"_ZStplIcSt11char_traitsIcESaIcEESbIT_T0_T1_EOS6_PKS3_"}
!67 = !{!68, !14, i64 0}
!68 = !{!"_ZTSN2cv5Size_IiEE", !14, i64 0, !14, i64 4}
!69 = !{!68, !14, i64 4}
!70 = !{!71, !14, i64 0}
!71 = !{!"_ZTSN2cv11_InputArrayE", !14, i64 0, !3, i64 8, !68, i64 16}
!72 = !{!71, !3, i64 8}
!73 = !{!74, !3, i64 0}
!74 = !{!"_ZTSSt12_Vector_baseIiSaIiEE", !75, i64 0}
!75 = !{!"_ZTSNSt12_Vector_baseIiSaIiEE12_Vector_implE", !3, i64 0, !3, i64 8, !3, i64 16}
!76 = !{!17, !3, i64 56}
!77 = !{!17, !3, i64 64}
!78 = !{!17, !3, i64 48}
!79 = !{!80}
!80 = distinct !{!80, !81, !"_ZStplIcSt11char_traitsIcESaIcEESbIT_T0_T1_ERKS6_PKS3_: %agg.result"}
!81 = distinct !{!81, !"_ZStplIcSt11char_traitsIcESaIcEESbIT_T0_T1_ERKS6_PKS3_"}
!82 = !{!83}
!83 = distinct !{!83, !84, !"_ZStplIcSt11char_traitsIcESaIcEESbIT_T0_T1_EOS6_S7_: %agg.result"}
!84 = distinct !{!84, !"_ZStplIcSt11char_traitsIcESaIcEESbIT_T0_T1_EOS6_S7_"}
!85 = !{!86}
!86 = distinct !{!86, !87, !"_ZStplIcSt11char_traitsIcESaIcEESbIT_T0_T1_EOS6_PKS3_: %agg.result"}
!87 = distinct !{!87, !"_ZStplIcSt11char_traitsIcESaIcEESbIT_T0_T1_EOS6_PKS3_"}
