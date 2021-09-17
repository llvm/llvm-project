// These declarations have intentionally the same name as the function-local
// class. LLDB should never pull in these definitions as this test only inspects
// the classes defined in the function below.
struct WithMember {
  float false_def;
};
typedef struct {
  float false_def;
} TypedefUnnamed;
struct ForwardConflict {
  float false_def;
};
ForwardConflict conflict1;
WithMember conflict2;
struct {
  float false_def;
} unnamed;

int main() {
  struct WithMember {
    int i;
  };
  typedef struct {
    int a;
  } TypedefUnnamed;
  typedef struct {
    int b;
  } TypedefUnnamed2;
  struct Forward;
  struct ForwardConflict;

  WithMember m = {1};
  TypedefUnnamed typedef_unnamed = {2};
  TypedefUnnamed2 typedef_unnamed2 = {3};
  struct {
    int i;
  } unnamed = {4};
  struct {
    int j;
  } unnamed2 = {5};
  Forward *fwd = nullptr;
  ForwardConflict *fwd_conflict = nullptr;
  return 0; // break here
}
