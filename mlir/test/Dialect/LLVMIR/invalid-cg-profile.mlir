// RUN: mlir-translate %s -mlir-to-llvmir | FileCheck %s
// CHECK: !llvm.module.flags

module {
  llvm.module_flags [#llvm.mlir.module_flag<error, "wchar_size", 4 : i32>,
                     #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>,
                     #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>,
                     #llvm.mlir.module_flag<max, "uwtable", 2 : i32>,
                     #llvm.mlir.module_flag<max, "frame-pointer", 1 : i32>,
                     #llvm.mlir.module_flag<override, "probe-stack", "inline-asm">,
                     #llvm.mlir.module_flag<append, "CG Profile", [
                       #llvm.cgprofile_entry<from = @from, to = @to, count = 222>,
                       #llvm.cgprofile_entry<from = @from, count = 222>,
                       #llvm.cgprofile_entry<from = @to, to = @from, count = 222>
                    ]>,
                    #llvm.mlir.module_flag<error, "ProfileSummary",
                       #llvm.profile_summary<format = InstrProf, total_count = 263646, max_count = 86427,
                         max_internal_count = 86427, max_function_count = 4691,
                         num_counts = 3712, num_functions = 796,
                         is_partial_profile = 0,
                         partial_profile_ratio = 0.000000e+00 : f64,
                         detailed_summary =
                           <cut_off = 10000, min_count = 86427, num_counts = 1>,
                           <cut_off = 100000, min_count = 86427, num_counts = 1>
                    >>]
}
