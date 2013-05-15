// Some convenient things to return:
static char *g_first_pointer = "I am the first";
static char *g_second_pointer = "I am the second";

// First we have some simple functions that return standard types, ints, floats and doubles.
// We have a function calling a function in a few cases to test that if you stop in the
// inner function then do "up/fin" you get the return value from the outer-most frame.

int 
inner_sint (int value)
{
  return value;
}

int
outer_sint (int value)
{
  int outer_value = 2 * inner_sint (value);
  return outer_value;
}

float
inner_float (float value)
{
  return value;
}

float 
outer_float (float value)
{
  float outer_value = 2 * inner_float(value);
  return outer_value;
}

double 
return_double (double value)
{
  return value;
}

long double 
return_long_double (long double value)
{
  return value;
}

char *
return_pointer (char *value)
{
  return value;
}

struct one_int
{
  int one_field;
};

struct one_int
return_one_int (struct one_int value)
{
  return value;
}

struct two_int
{
  int first_field;
  int second_field;
};

struct two_int
return_two_int (struct two_int value)
{
  return value;
}

struct three_int
{
  int first_field;
  int second_field;
  int third_field;
};

struct three_int
return_three_int (struct three_int value)
{
  return value;
}

struct four_int
{
  int first_field;
  int second_field;
  int third_field;
  int fourth_field;
};

struct four_int
return_four_int (struct four_int value)
{
  return value;
}

struct five_int
{
  int first_field;
  int second_field;
  int third_field;
  int fourth_field;
  int fifth_field;
};

struct five_int
return_five_int (struct five_int value)
{
  return value;
}

struct one_int_one_double
{
  int first_field;
  double second_field;
};

struct one_int_one_double
return_one_int_one_double (struct one_int_one_double value)
{
  return value;
}

struct one_int_one_double_one_int
{
  int one_field;
  double second_field;
  int third_field;
};

struct one_int_one_double_one_int
return_one_int_one_double_one_int (struct one_int_one_double_one_int value)
{
  return value;
}

struct one_short_one_double_one_short
{
  int one_field;
  double second_field;
  int third_field;
};

struct one_short_one_double_one_short
return_one_short_one_double_one_short (struct one_short_one_double_one_short value)
{
  return value;
}

struct three_short_one_float
{
  short one_field;
  short second_field;
  short third_field;
  float fourth_field;
};

struct three_short_one_float
return_three_short_one_float (struct three_short_one_float value)
{
  return value;
}

struct one_int_one_float_one_int
{
  int one_field;
  float second_field;
  int third_field;
};

struct one_int_one_float_one_int
return_one_int_one_float_one_int (struct one_int_one_float_one_int value)
{
  return value;
}

struct one_float_one_int_one_float
{
  float one_field;
  int second_field;
  float third_field;
};

struct one_float_one_int_one_float
return_one_float_one_int_one_float (struct one_float_one_int_one_float value)
{
  return value;
}

struct one_double_two_float
{
  double one_field;
  float second_field;
  float third_field;
};

struct one_double_two_float
return_one_double_two_float (struct one_double_two_float value)
{
  return value;
}

struct two_double
{
  double first_field;
  double second_field;
};

struct two_double
return_two_double (struct two_double value)
{
  return value;
}

struct two_float
{
  float first_field;
  float second_field;
};

struct two_float
return_two_float (struct two_float value)
{
  return value;
}

struct one_int_one_double_packed
{
  int first_field;
  double second_field;
} __attribute__((__packed__));

struct one_int_one_double_packed
return_one_int_one_double_packed (struct one_int_one_double_packed value)
{
  return value;
}

struct one_int_one_long
{
  int first_field;
  long second_field;
};

struct one_int_one_long
return_one_int_one_long (struct one_int_one_long value)
{
  return value;
}

struct one_pointer
{
  char *first_field;
};

struct one_pointer
return_one_pointer (struct one_pointer value)
{
  return value;
}

struct two_pointer
{
  char *first_field;
  char *second_field;
};

struct two_pointer
return_two_pointer (struct two_pointer value)
{
  return value;
}

struct one_float_one_pointer
{
  float first_field;
  char *second_field;
};

struct one_float_one_pointer
return_one_float_one_pointer (struct one_float_one_pointer value)
{
  return value;
}

struct one_int_one_pointer
{
  int first_field;
  char *second_field;
};

struct one_int_one_pointer
return_one_int_one_pointer (struct one_int_one_pointer value)
{
  return value;
}

typedef float vector_size_float32 __attribute__((__vector_size__(16)));
typedef float ext_vector_size_float32 __attribute__((ext_vector_type(4)));

vector_size_float32
return_vector_size_float32 (vector_size_float32 value)
{
    return value;
}

ext_vector_size_float32
return_ext_vector_size_float32 (ext_vector_size_float32 value)
{
    return value;
}

int 
main ()
{
  int first_int = 123456;
  int second_int = 234567;

  outer_sint (first_int);
  outer_sint (second_int);

  float first_float_value = 12.34;
  float second_float_value = 23.45;

  outer_float (first_float_value);
  outer_float (second_float_value);

  double double_value = -23.45;

  return_double (double_value);

  return_pointer(g_first_pointer);

  long double long_double_value = -3456789.987654321;

  return_long_double (long_double_value);

  // Okay, now the structures:
  return_one_int ((struct one_int) {10});
  return_two_int ((struct two_int) {10, 20});
  return_three_int ((struct three_int) {10, 20, 30});
  return_four_int ((struct four_int) {10, 20, 30, 40});
  return_five_int ((struct five_int) {10, 20, 30, 40, 50});

  return_two_double ((struct two_double) {10.0, 20.0});
  return_one_double_two_float ((struct one_double_two_float) {10.0, 20.0, 30.0});
  return_one_int_one_float_one_int ((struct one_int_one_float_one_int) {10, 20.0, 30});

  return_one_pointer ((struct one_pointer) {g_first_pointer});
  return_two_pointer ((struct two_pointer) {g_first_pointer, g_second_pointer});
  return_one_float_one_pointer ((struct one_float_one_pointer) {10.0, g_first_pointer});
  return_one_int_one_pointer ((struct one_int_one_pointer) {10, g_first_pointer});
  return_three_short_one_float ((struct three_short_one_float) {10, 20, 30, 40.0});

  return_one_int_one_double ((struct one_int_one_double) {10, 20.0});
  return_one_int_one_double_one_int ((struct one_int_one_double_one_int) {10, 20.0, 30});
  return_one_short_one_double_one_short ((struct one_short_one_double_one_short) {10, 20.0, 30});
  return_one_float_one_int_one_float ((struct one_float_one_int_one_float) {10.0, 20, 30.0});
  return_two_float ((struct two_float) { 10.0, 20.0});
  return_one_int_one_double_packed ((struct one_int_one_double_packed) {10, 20.0});
  return_one_int_one_long ((struct one_int_one_long) {10, 20});

  return_vector_size_float32 (( vector_size_float32 ){1.5, 2.25, 4.125, 8.0625});
  return_ext_vector_size_float32 ((ext_vector_size_float32){ 16.5, 32.25, 64.125, 128.0625});
  
  return 0;
  
}
