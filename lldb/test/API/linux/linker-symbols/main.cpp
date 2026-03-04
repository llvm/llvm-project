extern "C" {

extern void text_func() {}
[[gnu::section("pseudo_text")]] extern void pseudo_text_func() {}

extern int data_var;
int data_var = 1;
extern int pseudo_data_var;
[[gnu::section("pseudo_data")]] int pseudo_data_var = 1;

extern int bss_var;
int bss_var;
extern int pseudo_bss_var;
[[gnu::section("pseudo_bss")]] int pseudo_bss_var;

}

int main(int argc, char **argv) {
  return 0; // Set a breakpoint here
}
