extern "C" int b_function();

extern "C" int a_function() { return b_function(); }
