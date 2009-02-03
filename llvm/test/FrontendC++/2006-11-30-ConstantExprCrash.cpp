// RUN: %llvmgxx %s -emit-llvm -S -o -
// PR1027

struct sys_var {
  unsigned name_length;

  bool no_support_one_shot;
  sys_var() {}
};


struct sys_var_thd : public sys_var {
};

extern sys_var_thd sys_auto_is_null;

sys_var *getsys_variables() {
  return &sys_auto_is_null;
}

sys_var *sys_variables = &sys_auto_is_null;






