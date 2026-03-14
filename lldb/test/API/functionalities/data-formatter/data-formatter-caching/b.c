typedef float float4 __attribute__((ext_vector_type(4)));
void stop() {}
int a() {
  float4 f4 = {1, 2, 3, 4};
  // break here
  stop();
  return 0;
}
