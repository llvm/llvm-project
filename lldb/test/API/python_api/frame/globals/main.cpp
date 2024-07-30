class Vars {
public:
  inline static double inline_static = 1.5;
  static constexpr int static_constexpr = 2;
  static const int static_const_out_out_class;
};

const int Vars::static_const_out_out_class = 3;

char global_var_of_char_type = 'X';

int main() {}
