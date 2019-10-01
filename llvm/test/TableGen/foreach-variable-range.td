// RUN: llvm-tblgen %s

class ConstantsImpl {
  int Zero = 0;
  int One = 1;
  int Two = 2;
  int Three = 3;
  int Five = 5;
}

def Constants : ConstantsImpl;

// CHECK-DAG: def var_bound_whitespaceA0
// CHECK-DAG: def var_bound_whitespaceA1
// CHECK-DAG: def var_bound_whitespaceA2
foreach Index = Constants.Zero - Constants.Two in {
  def var_bound_whitespaceA#Index;
}

// CHECK-DAG: def var_bound_whitespaceB0
// CHECK-DAG: def var_bound_whitespaceB1
// CHECK-DAG: def var_bound_whitespaceB2
foreach Index = Constants.Zero-Constants.Two in {
  def var_bounds_whitespaceB#Index;
}

// CHECK-DAG: def var_bound_whitespaceC0
// CHECK-DAG: def var_bound_whitespaceC1
// CHECK-DAG: def var_bound_whitespaceC2
foreach Index = Constants.Zero -Constants.Two in {
  def var_bounds_whitespaceC#Index;
}

// CHECK-DAG: def var_bound_whitespaceD0
// CHECK-DAG: def var_bound_whitespaceD1
// CHECK-DAG: def var_bound_whitespaceD2
foreach Index = Constants.Zero- Constants.Two in {
  def var_bounds_whitespaceD#Index;
}

// CHECK-DAG: def const_lower_whitespaceA0
// CHECK-DAG: def const_lower_whitespaceA1
// CHECK-DAG: def const_lower_whitespaceA2
foreach Index = 0 - Constants.Two in {
  def const_lower_whitespaceA#Index;
}

// CHECK-DAG: def const_lower_whitespaceB0
// CHECK-DAG: def const_lower_whitespaceB1
// CHECK-DAG: def const_lower_whitespaceB2
foreach Index = 0-Constants.Two in {
  def const_lower_whitespaceB#Index;
}

// CHECK-DAG: def const_lower_whitespaceC0
// CHECK-DAG: def const_lower_whitespaceC1
// CHECK-DAG: def const_lower_whitespaceC2
foreach Index = 0 -Constants.Two in {
  def const_lower_whitespaceC#Index;
}

// CHECK-DAG: def const_lower_whitespaceD0
// CHECK-DAG: def const_lower_whitespaceD1
// CHECK-DAG: def const_lower_whitespaceD2
foreach Index = 0- Constants.Two in {
  def const_lower_whitespaceD#Index;
}

// CHECK-DAG: def const_upper_whitespaceA0
// CHECK-DAG: def const_upper_whitespaceA1
// CHECK-DAG: def const_upper_whitespaceA2
foreach Index = Constants.Zero - 2 in {
  def const_upper_whitespaceA#Index;
}

// CHECK-DAG: def const_upper_whitespaceB0
// CHECK-DAG: def const_upper_whitespaceB1
// CHECK-DAG: def const_upper_whitespaceB2
foreach Index = Constants.Zero-2 in {
  def const_upper_whitespaceB#Index;
}

// CHECK-DAG: def const_upper_whitespaceC0
// CHECK-DAG: def const_upper_whitespaceC1
// CHECK-DAG: def const_upper_whitespaceC2
foreach Index = Constants.Zero -2 in {
  def const_upper_whitespaceC#Index;
}

// CHECK-DAG: def const_upper_whitespaceD0
// CHECK-DAG: def const_upper_whitespaceD1
// CHECK-DAG: def const_upper_whitespaceD2
foreach Index = Constants.Zero- 2 in {
  def const_upper_whitespaceD#Index;
}

// CHECK-DAG: def multi_rangeA0
// CHECK-DAG: def multi_rangeA1
// CHECK-DAG: def multi_rangeA2
// CHECK-DAG: def multi_rangeA3
foreach Index = {Constants.Zero-Constants.One, Constants.Two-Constants.Three} in {
  def multi_rangeA#Index;
}

// CHECK-DAG: def multi_rangeB0
// CHECK-DAG: def multi_rangeB1
// CHECK-DAG: def multi_rangeB3
// CHECK-DAG: def multi_rangeB4
// CHECK-DAG: def multi_rangeB5
foreach Index = {0-Constants.One, Constants.Three-Constants.Five} in {
  def multi_rangeB#Index;
}

// CHECK-DAG: def multi_rangeC0
// CHECK-DAG: def multi_rangeC1
// CHECK-DAG: def multi_rangeC2
// CHECK-DAG: def multi_rangeC3
foreach Index = {0-Constants.One, 2-Constants.Three} in {
  def multi_rangeC#Index;
}

// CHECK-DAG: def multi_rangeD0
// CHECK-DAG: def multi_rangeD1
// CHECK-DAG: def multi_rangeD2
// CHECK-DAG: def multi_rangeD3
foreach Index = {0-1, Constants.Two-3} in {
  def multi_rangeD#Index;
}
