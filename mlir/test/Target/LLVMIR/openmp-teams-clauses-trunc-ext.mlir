// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 22.0.0 (git@github.com:mjklemm/llvm-project.git 5d9164c24a474793ab325116c5f782dce0577574)", llvm.target_triple = "x86_64-unknown-linux-gnu", omp.is_gpu = false, omp.is_target_device = false, omp.requires = #omp<clause_requires none>, omp.target_triples = [], omp.version = #omp.version<version = 31>} {
  omp.private {type = private} @_QFnum_threads_arg_2Ei_private_i32 : i32
  omp.private {type = private} @_QFnum_threads_const_2Ei_private_i32 : i32
  omp.private {type = private} @_QFnum_threads_arg_8Ei_private_i32 : i32
  omp.private {type = private} @_QFnum_threads_const_8Ei_private_i32 : i32
  omp.private {type = private} @_QFthread_limit_arg_2Ei_private_i32 : i32
  omp.private {type = private} @_QFthread_limit_const_2Ei_private_i32 : i32
  omp.private {type = private} @_QFthread_limit_arg_8Ei_private_i32 : i32
  omp.private {type = private} @_QFthread_limit_const_8Ei_private_i32 : i32
  omp.private {type = private} @_QFnum_teams_arg_2Ei_private_i32 : i32
  omp.private {type = private} @_QFnum_teams_const_2Ei_private_i32 : i32
  omp.private {type = private} @_QFnum_teams_arg_8Ei_private_i32 : i32
  omp.private {type = private} @_QFnum_teams_const_8Ei_private_i32 : i32

  llvm.func @_QPnum_teams_const_8(%arg0: !llvm.ptr {fir.bindc_name = "n", llvm.noalias, llvm.nocapture}) {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(137 : i64) : i64
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.alloca %2 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
    %4 = llvm.load %arg0 : !llvm.ptr -> i32
    %5 = omp.map.info var_ptr(%3 : !llvm.ptr, i32) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr {name = "i"}
    %6 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr {name = "n"}
    omp.target host_eval(%0 -> %arg1, %4 -> %arg2, %0 -> %arg3, %1 -> %arg4 : i32, i32, i32, i64) map_entries(%5 -> %arg5, %6 -> %arg6 : !llvm.ptr, !llvm.ptr) {
      omp.teams num_teams( to %arg4 : i64) {
        omp.distribute private(@_QFnum_teams_const_8Ei_private_i32 %arg5 -> %arg7 : !llvm.ptr) {
          omp.loop_nest (%arg8) : i32 = (%arg1) to (%arg2) inclusive step (%arg3) {
            llvm.store %arg8, %arg7 : i32, !llvm.ptr
            omp.yield
          }
        }
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }
  // CHECK: @__omp_offloading_{{.*}}_{{.*}}__QPnum_teams_const_8_l{{.*}}(i32 %{{.*}}, ptr %{{.*}}, ptr %{{.*}})
  // CHECK: call void @__kmpc_push_num_teams_51({{.*}}, {{.*}}, i32 137, i32 137, {{.*}})

  llvm.func @_QPnum_teams_arg_8(%arg0: !llvm.ptr {fir.bindc_name = "n", llvm.noalias, llvm.nocapture}, %arg1: !llvm.ptr {fir.bindc_name = "t", llvm.noalias, llvm.nocapture}) {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.alloca %1 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
    %3 = llvm.load %arg0 : !llvm.ptr -> i32
    %4 = llvm.load %arg1 : !llvm.ptr -> i64
    %5 = omp.map.info var_ptr(%arg1 : !llvm.ptr, i64) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr {name = "t"}
    %6 = omp.map.info var_ptr(%2 : !llvm.ptr, i32) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr {name = "i"}
    %7 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr {name = "n"}
    omp.target host_eval(%0 -> %arg2, %3 -> %arg3, %0 -> %arg4, %4 -> %arg5 : i32, i32, i32, i64) map_entries(%5 -> %arg6, %6 -> %arg7, %7 -> %arg8 : !llvm.ptr, !llvm.ptr, !llvm.ptr) {
      omp.teams num_teams( to %arg5 : i64) {
        omp.distribute private(@_QFnum_teams_arg_8Ei_private_i32 %arg7 -> %arg9 : !llvm.ptr) {
          omp.loop_nest (%arg10) : i32 = (%arg2) to (%arg3) inclusive step (%arg4) {
            llvm.store %arg10, %arg9 : i32, !llvm.ptr
            omp.yield
          }
        }
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }
  // CHECK: @__omp_offloading_{{.*}}_{{.*}}__QPnum_teams_arg_8_l{{.*}}(i32 %{{.*}}, i64 %[[ARG:.*]], ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}})
  // CHECK: %[[CONV_LB:.*]] = trunc i64 %[[ARG]] to i32
  // CHECK: %[[CONV_UB:.*]] = trunc i64 %[[ARG]] to i32
  // CHECK: call void @__kmpc_push_num_teams_51({{.*}}, {{.*}}, i32 %[[CONV_LB]], i32 %[[CONV_UB]], {{.*}})

  llvm.func @_QPnum_teams_const_2(%arg0: !llvm.ptr {fir.bindc_name = "n", llvm.noalias, llvm.nocapture}) {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(137 : i16) : i16
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.alloca %2 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
    %4 = llvm.load %arg0 : !llvm.ptr -> i32
    %5 = omp.map.info var_ptr(%3 : !llvm.ptr, i32) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr {name = "i"}
    %6 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr {name = "n"}
    omp.target host_eval(%0 -> %arg1, %4 -> %arg2, %0 -> %arg3, %1 -> %arg4 : i32, i32, i32, i16) map_entries(%5 -> %arg5, %6 -> %arg6 : !llvm.ptr, !llvm.ptr) {
      omp.teams num_teams( to %arg4 : i16) {
        omp.distribute private(@_QFnum_teams_const_2Ei_private_i32 %arg5 -> %arg7 : !llvm.ptr) {
          omp.loop_nest (%arg8) : i32 = (%arg1) to (%arg2) inclusive step (%arg3) {
            llvm.store %arg8, %arg7 : i32, !llvm.ptr
            omp.yield
          }
        }
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }
  // CHECK: @__omp_offloading_{{.*}}_{{.*}}__QPnum_teams_const_2_l{{.*}}(i32 %{{.*}}, ptr %{{.*}}, ptr %{{.*}})
  // CHECK: call void @__kmpc_push_num_teams_51({{.*}}, {{.*}}, i32 137, i32 137, {{.*}})

  llvm.func @_QPnum_teams_arg_2(%arg0: !llvm.ptr {fir.bindc_name = "n", llvm.noalias, llvm.nocapture}, %arg1: !llvm.ptr {fir.bindc_name = "t", llvm.noalias, llvm.nocapture}) {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.alloca %1 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
    %3 = llvm.load %arg0 : !llvm.ptr -> i32
    %4 = llvm.load %arg1 : !llvm.ptr -> i16
    %5 = omp.map.info var_ptr(%arg1 : !llvm.ptr, i16) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr {name = "t"}
    %6 = omp.map.info var_ptr(%2 : !llvm.ptr, i32) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr {name = "i"}
    %7 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr {name = "n"}
    omp.target host_eval(%0 -> %arg2, %3 -> %arg3, %0 -> %arg4, %4 -> %arg5 : i32, i32, i32, i16) map_entries(%5 -> %arg6, %6 -> %arg7, %7 -> %arg8 : !llvm.ptr, !llvm.ptr, !llvm.ptr) {
      omp.teams num_teams( to %arg5 : i16) {
        omp.distribute private(@_QFnum_teams_arg_2Ei_private_i32 %arg7 -> %arg9 : !llvm.ptr) {
          omp.loop_nest (%arg10) : i32 = (%arg2) to (%arg3) inclusive step (%arg4) {
            llvm.store %arg10, %arg9 : i32, !llvm.ptr
            omp.yield
          }
        }
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }
  // CHECK: @__omp_offloading_{{.*}}_{{.*}}__QPnum_teams_arg_2_l{{.*}}(i32 %{{.*}}, ptr %{{.*}}, ptr %{{.*}})
  // CHECK: %[[CONV_LB:.*]] = sext i16 %[[ARG]] to i32
  // CHECK: %[[CONV_UB:.*]] = sext i16 %[[ARG]] to i32
  // CHECK: call void @__kmpc_push_num_teams_51({{.*}}, {{.*}}, i32 %[[CONV_LB]], i32 %[[CONV_UB]], {{.*}})

  llvm.func @_QPthread_limit_const_8(%arg0: !llvm.ptr {fir.bindc_name = "n", llvm.noalias, llvm.nocapture}) {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.alloca %1 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
    %3 = llvm.load %arg0 : !llvm.ptr -> i32
    %4 = omp.map.info var_ptr(%2 : !llvm.ptr, i32) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr {name = "i"}
    %5 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr {name = "n"}
    omp.target host_eval(%0 -> %arg1, %3 -> %arg2, %0 -> %arg3 : i32, i32, i32) map_entries(%4 -> %arg4, %5 -> %arg5 : !llvm.ptr, !llvm.ptr) {
      %6 = llvm.mlir.constant(137 : i64) : i64
      omp.teams thread_limit(%6 : i64) {
        omp.distribute private(@_QFthread_limit_const_8Ei_private_i32 %arg4 -> %arg6 : !llvm.ptr) {
          omp.loop_nest (%arg7) : i32 = (%arg1) to (%arg2) inclusive step (%arg3) {
            llvm.store %arg7, %arg6 : i32, !llvm.ptr
            omp.yield
          }
        }
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }
  // CHECK: @__omp_offloading_{{.*}}_{{.*}}__QPthread_limit_const_8_l{{.*}}(i32 %{{.*}}, ptr %{{.*}}, ptr %{{.*}})
  // CHECK: call void @__kmpc_push_num_teams_51({{.*}}, {{.*}}, i32 0, i32 0, i32 137)

  llvm.func @_QPthread_limit_arg_8(%arg0: !llvm.ptr {fir.bindc_name = "n", llvm.noalias, llvm.nocapture}, %arg1: !llvm.ptr {fir.bindc_name = "t", llvm.noalias, llvm.nocapture}) {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.alloca %1 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
    %3 = llvm.load %arg0 : !llvm.ptr -> i32
    %4 = omp.map.info var_ptr(%arg1 : !llvm.ptr, i64) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr {name = "t"}
    %5 = omp.map.info var_ptr(%2 : !llvm.ptr, i32) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr {name = "i"}
    %6 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr {name = "n"}
    omp.target host_eval(%0 -> %arg2, %3 -> %arg3, %0 -> %arg4 : i32, i32, i32) map_entries(%4 -> %arg5, %5 -> %arg6, %6 -> %arg7 : !llvm.ptr, !llvm.ptr, !llvm.ptr) {
      %7 = llvm.load %arg5 : !llvm.ptr -> i64
      omp.teams thread_limit(%7 : i64) {
        omp.distribute private(@_QFthread_limit_arg_8Ei_private_i32 %arg6 -> %arg8 : !llvm.ptr) {
          omp.loop_nest (%arg9) : i32 = (%arg2) to (%arg3) inclusive step (%arg4) {
            llvm.store %arg9, %arg8 : i32, !llvm.ptr
            omp.yield
          }
        }
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }
  // CHECK: @__omp_offloading_{{.*}}_{{.*}}__QPthread_limit_arg_8_l{{.*}}(i32 %{{.*}}, ptr %[[ARG:.*]], ptr %{{.*}}, ptr %{{.*}})
  // CHECK: %[[ARG_LD:.*]] = load i64, ptr %[[ARG]], align 8
  // CHECK: %[[CONV_TL:.*]] = trunc i64 %[[ARG_LD]] to i32
  // CHECK: call void @__kmpc_push_num_teams_51({{.*}}, {{.*}}, i32 0, i32 0, i32 %[[CONV_TL]])

  llvm.func @_QPthread_limit_const_2(%arg0: !llvm.ptr {fir.bindc_name = "n", llvm.noalias, llvm.nocapture}) {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.alloca %1 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
    %3 = llvm.load %arg0 : !llvm.ptr -> i32
    %4 = omp.map.info var_ptr(%2 : !llvm.ptr, i32) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr {name = "i"}
    %5 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr {name = "n"}
    omp.target host_eval(%0 -> %arg1, %3 -> %arg2, %0 -> %arg3 : i32, i32, i32) map_entries(%4 -> %arg4, %5 -> %arg5 : !llvm.ptr, !llvm.ptr) {
      %6 = llvm.mlir.constant(137 : i16) : i16
      omp.teams thread_limit(%6 : i16) {
        omp.distribute private(@_QFthread_limit_const_2Ei_private_i32 %arg4 -> %arg6 : !llvm.ptr) {
          omp.loop_nest (%arg7) : i32 = (%arg1) to (%arg2) inclusive step (%arg3) {
            llvm.store %arg7, %arg6 : i32, !llvm.ptr
            omp.yield
          }
        }
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }
  // CHECK: @__omp_offloading_{{.*}}_{{.*}}__QPthread_limit_const_2_l{{.*}}(i32 %{{.*}}, ptr %{{.*}}, ptr %{{.*}})
  // CHECK: call void @__kmpc_push_num_teams_51({{.*}}, {{.*}}, i32 0, i32 0, i32 137)

  llvm.func @_QPthread_limit_arg_2(%arg0: !llvm.ptr {fir.bindc_name = "n", llvm.noalias, llvm.nocapture}, %arg1: !llvm.ptr {fir.bindc_name = "t", llvm.noalias, llvm.nocapture}) {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.alloca %1 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
    %3 = llvm.load %arg0 : !llvm.ptr -> i32
    %4 = omp.map.info var_ptr(%arg1 : !llvm.ptr, i16) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr {name = "t"}
    %5 = omp.map.info var_ptr(%2 : !llvm.ptr, i32) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr {name = "i"}
    %6 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr {name = "n"}
    omp.target host_eval(%0 -> %arg2, %3 -> %arg3, %0 -> %arg4 : i32, i32, i32) map_entries(%4 -> %arg5, %5 -> %arg6, %6 -> %arg7 : !llvm.ptr, !llvm.ptr, !llvm.ptr) {
      %7 = llvm.load %arg5 : !llvm.ptr -> i16
      omp.teams thread_limit(%7 : i16) {
        omp.distribute private(@_QFthread_limit_arg_2Ei_private_i32 %arg6 -> %arg8 : !llvm.ptr) {
          omp.loop_nest (%arg9) : i32 = (%arg2) to (%arg3) inclusive step (%arg4) {
            llvm.store %arg9, %arg8 : i32, !llvm.ptr
            omp.yield
          }
        }
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }
  // CHECK: @__omp_offloading_{{.*}}_{{.*}}__QPthread_limit_arg_2_l{{.*}}(i32 %{{.*}}, ptr %[[ARG:.*]], ptr %{{.*}}, ptr %{{.*}})
  // CHECK: %[[ARG_LD:.*]] = load i16, ptr %[[ARG]], align 2
  // CHECK: %[[CONV_TL:.*]] = sext i16 %[[ARG_LD]] to i32
  // CHECK: call void @__kmpc_push_num_teams_51({{.*}}, {{.*}}, i32 0, i32 0, i32 %[[CONV_TL]])

  llvm.func @_QPnum_threads_const_8(%arg0: !llvm.ptr {fir.bindc_name = "n", llvm.noalias, llvm.nocapture}) {
    %0 = llvm.mlir.constant(137 : i64) : i64
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.alloca %2 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
    %4 = llvm.load %arg0 : !llvm.ptr -> i32
    %5 = omp.map.info var_ptr(%3 : !llvm.ptr, i32) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr {name = "i"}
    %6 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr {name = "n"}
    omp.target host_eval(%1 -> %arg1, %4 -> %arg2, %1 -> %arg3, %0 -> %arg4 : i32, i32, i32, i64) map_entries(%5 -> %arg5, %6 -> %arg6 : !llvm.ptr, !llvm.ptr) {
      omp.teams {
        omp.parallel num_threads(%arg4 : i64) private(@_QFnum_threads_const_8Ei_private_i32 %arg5 -> %arg7 : !llvm.ptr) {
          omp.distribute {
            omp.wsloop {
              omp.loop_nest (%arg8) : i32 = (%arg1) to (%arg2) inclusive step (%arg3) {
                llvm.store %arg8, %arg7 : i32, !llvm.ptr
                omp.yield
              }
            } {omp.composite}
          } {omp.composite}
          omp.terminator
        } {omp.composite}
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }
  // CHECK: @__omp_offloading_{{.*}}_{{.*}}__QPnum_threads_const_8_l{{.*}}..omp_par.{{.*}}
  // CHECK: call void @__kmpc_push_num_threads({{.*}}, {{.*}}, i32 137)

  llvm.func @_QPnum_threads_arg_8(%arg0: !llvm.ptr {fir.bindc_name = "n", llvm.noalias, llvm.nocapture}) {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.alloca %1 x i64 {bindc_name = "t"} : (i64) -> !llvm.ptr
    %3 = llvm.alloca %1 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
    %4 = llvm.load %2 : !llvm.ptr -> i64
    %5 = llvm.load %arg0 : !llvm.ptr -> i32
    %6 = omp.map.info var_ptr(%2 : !llvm.ptr, i64) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr {name = "t"}
    %7 = omp.map.info var_ptr(%3 : !llvm.ptr, i32) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr {name = "i"}
    %8 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr {name = "n"}
    omp.target host_eval(%0 -> %arg1, %5 -> %arg2, %0 -> %arg3, %4 -> %arg4 : i32, i32, i32, i64) map_entries(%6 -> %arg5, %7 -> %arg6, %8 -> %arg7 : !llvm.ptr, !llvm.ptr, !llvm.ptr) {
      omp.teams {
        omp.parallel num_threads(%arg4 : i64) private(@_QFnum_threads_arg_8Ei_private_i32 %arg6 -> %arg8 : !llvm.ptr) {
          omp.distribute {
            omp.wsloop {
              omp.loop_nest (%arg9) : i32 = (%arg1) to (%arg2) inclusive step (%arg3) {
                llvm.store %arg9, %arg8 : i32, !llvm.ptr
                omp.yield
              }
            } {omp.composite}
          } {omp.composite}
          omp.terminator
        } {omp.composite}
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }
  // CHECK: @__omp_offloading_{{.*}}_{{.*}}__QPnum_threads_arg_8_l{{.*}}..omp_par.{{.*}}
  // CHECK: %[[CONV_NT:.*]] = trunc i64 %loadgep_ to i32
  // CHECK: call void @__kmpc_push_num_threads({{.*}}, {{.*}}, i32 %[[CONV_NT]])

  llvm.func @_QPnum_threads_const_2(%arg0: !llvm.ptr {fir.bindc_name = "n", llvm.noalias, llvm.nocapture}) {
    %0 = llvm.mlir.constant(137 : i16) : i16
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.alloca %2 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
    %4 = llvm.load %arg0 : !llvm.ptr -> i32
    %5 = omp.map.info var_ptr(%3 : !llvm.ptr, i32) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr {name = "i"}
    %6 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr {name = "n"}
    omp.target host_eval(%1 -> %arg1, %4 -> %arg2, %1 -> %arg3, %0 -> %arg4 : i32, i32, i32, i16) map_entries(%5 -> %arg5, %6 -> %arg6 : !llvm.ptr, !llvm.ptr) {
      omp.teams {
        omp.parallel num_threads(%arg4 : i16) private(@_QFnum_threads_const_2Ei_private_i32 %arg5 -> %arg7 : !llvm.ptr) {
          omp.distribute {
            omp.wsloop {
              omp.loop_nest (%arg8) : i32 = (%arg1) to (%arg2) inclusive step (%arg3) {
                llvm.store %arg8, %arg7 : i32, !llvm.ptr
                omp.yield
              }
            } {omp.composite}
          } {omp.composite}
          omp.terminator
        } {omp.composite}
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }
  // CHECK: @__omp_offloading_{{.*}}_{{.*}}__QPnum_threads_const_2_l{{.*}}..omp_par.{{.*}}
  // CHECK: call void @__kmpc_push_num_threads({{.*}}, {{.*}}, i32 137)

  llvm.func @_QPnum_threads_arg_2(%arg0: !llvm.ptr {fir.bindc_name = "n", llvm.noalias, llvm.nocapture}, %arg1: !llvm.ptr {fir.bindc_name = "t", llvm.noalias, llvm.nocapture}) {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.alloca %1 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
    %3 = llvm.load %arg1 : !llvm.ptr -> i16
    %4 = llvm.load %arg0 : !llvm.ptr -> i32
    %5 = omp.map.info var_ptr(%arg1 : !llvm.ptr, i16) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr {name = "t"}
    %6 = omp.map.info var_ptr(%2 : !llvm.ptr, i32) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr {name = "i"}
    %7 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(implicit) capture(ByCopy) -> !llvm.ptr {name = "n"}
    omp.target host_eval(%0 -> %arg2, %4 -> %arg3, %0 -> %arg4, %3 -> %arg5 : i32, i32, i32, i16) map_entries(%5 -> %arg6, %6 -> %arg7, %7 -> %arg8 : !llvm.ptr, !llvm.ptr, !llvm.ptr) {
      omp.teams {
        omp.parallel num_threads(%arg5 : i16) private(@_QFnum_threads_arg_2Ei_private_i32 %arg7 -> %arg9 : !llvm.ptr) {
          omp.distribute {
            omp.wsloop {
              omp.loop_nest (%arg10) : i32 = (%arg2) to (%arg3) inclusive step (%arg4) {
                llvm.store %arg10, %arg9 : i32, !llvm.ptr
                omp.yield
              }
            } {omp.composite}
          } {omp.composite}
          omp.terminator
        } {omp.composite}
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }
  // CHECK: @__omp_offloading_{{.*}}_{{.*}}__QPnum_threads_arg_2_l{{.*}}..omp_par.{{.*}}
  // CHECK: %[[CONV_NT:.*]] = zext i16 %loadgep_ to i32
  // CHECK: call void @__kmpc_push_num_threads({{.*}}, {{.*}}, i32 %[[CONV_NT]])
}
