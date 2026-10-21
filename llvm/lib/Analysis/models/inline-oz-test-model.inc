static const int64_t __constant_xi64 = 1;
class Model {
public:
  int64_t *fieldName0;
  int64_t *fieldName1;
  int64_t *fieldName2;
  int64_t *fieldName3;
  int64_t *fieldName4;
  int64_t *fieldName5;
  int64_t *fieldName6;
  int64_t *fieldName7;
  int64_t *fieldName8;
  int64_t *fieldName9;
  int64_t *fieldName10;
  int64_t *fieldName11;
  int64_t *fieldName12;
  int64_t *fieldName13;
  int64_t *fieldName14;
  int64_t *fieldName15;
  int64_t *fieldName16;
  int64_t *fieldName17;
  int64_t *fieldName18;
  int64_t *fieldName19;
  int64_t *fieldName20;
  int64_t *fieldName21;
  int64_t *fieldName22;
  int64_t *fieldName23;
  int64_t *fieldName24;
  int64_t *fieldName25;
  int64_t *fieldName26;
  int32_t *fieldName27;
  int64_t *fieldName28;
  int64_t *fieldName29;
  int64_t *fieldName30;
  int64_t *fieldName31;
  int64_t *fieldName32;
  int64_t *fieldName33;
  int64_t *fieldName34;
  int64_t *fieldName35;
  int64_t *fieldName36;
  float *fieldName37;
  int64_t *fieldName38;
  float *fieldName39;
  const std::map<std::string, char *> reflectionMap = {
      {"callsite_cost", reinterpret_cast<char *>(&fieldName0)},
      {"is_multiple_blocks", reinterpret_cast<char *>(&fieldName1)},
      {"caller_conditionally_executed_blocks",
       reinterpret_cast<char *>(&fieldName2)},
      {"cold_cc_penalty", reinterpret_cast<char *>(&fieldName3)},
      {"callee_conditionally_executed_blocks",
       reinterpret_cast<char *>(&fieldName4)},
      {"callee_users", reinterpret_cast<char *>(&fieldName5)},
      {"callee_basic_block_count", reinterpret_cast<char *>(&fieldName6)},
      {"nr_ctant_params", reinterpret_cast<char *>(&fieldName7)},
      {"load_relative_intrinsic", reinterpret_cast<char *>(&fieldName8)},
      {"jump_table_penalty", reinterpret_cast<char *>(&fieldName9)},
      {"unsimplified_common_instructions",
       reinterpret_cast<char *>(&fieldName10)},
      {"indirect_call_penalty", reinterpret_cast<char *>(&fieldName11)},
      {"load_elimination", reinterpret_cast<char *>(&fieldName12)},
      {"is_caller_avail_external", reinterpret_cast<char *>(&fieldName13)},
      {"call_penalty", reinterpret_cast<char *>(&fieldName14)},
      {"cost_estimate", reinterpret_cast<char *>(&fieldName15)},
      {"case_cluster_penalty", reinterpret_cast<char *>(&fieldName16)},
      {"node_count", reinterpret_cast<char *>(&fieldName17)},
      {"call_argument_setup", reinterpret_cast<char *>(&fieldName18)},
      {"sroa_savings", reinterpret_cast<char *>(&fieldName19)},
      {"lowered_call_arg_setup", reinterpret_cast<char *>(&fieldName20)},
      {"threshold", reinterpret_cast<char *>(&fieldName21)},
      {"dead_blocks", reinterpret_cast<char *>(&fieldName22)},
      {"constant_args", reinterpret_cast<char *>(&fieldName23)},
      {"sroa_losses", reinterpret_cast<char *>(&fieldName24)},
      {"simplified_instructions", reinterpret_cast<char *>(&fieldName25)},
      {"num_loops", reinterpret_cast<char *>(&fieldName26)},
      {"step_type", reinterpret_cast<char *>(&fieldName27)},
      {"edge_count", reinterpret_cast<char *>(&fieldName28)},
      {"is_callee_avail_external", reinterpret_cast<char *>(&fieldName29)},
      {"nested_inlines", reinterpret_cast<char *>(&fieldName30)},
      {"caller_basic_block_count", reinterpret_cast<char *>(&fieldName31)},
      {"last_call_to_static_bonus", reinterpret_cast<char *>(&fieldName32)},
      {"nested_inline_cost_estimate", reinterpret_cast<char *>(&fieldName33)},
      {"callsite_height", reinterpret_cast<char *>(&fieldName34)},
      {"constant_offset_ptr_args", reinterpret_cast<char *>(&fieldName35)},
      {"switch_penalty", reinterpret_cast<char *>(&fieldName36)},
      {"discount", reinterpret_cast<char *>(&fieldName37)},
      {"caller_users", reinterpret_cast<char *>(&fieldName38)},
      {"reward", reinterpret_cast<char *>(&fieldName39)}};
  char *getBufferForName(std::string v1) {
    char *v2 = reflectionMap.at(v1);
    return v2;
  }
  int64_t operator()() {
    int64_t v1 = 1;
    return v1;
  }
};
