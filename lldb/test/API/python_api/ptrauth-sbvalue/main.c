#define VALID_DATA_KEY 2

struct top {
  struct middle *unauth_middle;
  struct middle *__ptrauth(VALID_DATA_KEY) auth_middle;
};

struct middle {
  struct bottom *unauth_bottom;
  struct bottom *__ptrauth(VALID_DATA_KEY) auth_bottom;
};

struct bottom {
  int value;
  int *unauth_valuep;
  int *__ptrauth(VALID_DATA_KEY) auth_valuep;
};

int main(int argc, char **argv) {
  int stack_val = 2;
  struct bottom zero = {1, &stack_val, &stack_val};
  struct middle one = {&zero, &zero};
  struct top two = {&one, &one};

  int return_value = two.auth_middle->auth_bottom->value +
                     two.unauth_middle->unauth_bottom->value +
                     *two.auth_middle->auth_bottom->auth_valuep;

  return return_value; // break here
}
