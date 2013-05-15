#include <stdio.h>
#include <vector>
#include <string>

struct struct_for_copying
{
    struct_for_copying (int in_int, double in_double, const char *in_string) :
        int_value(in_int),
        double_value(in_double),
        string_value (in_string)
    {
        
    }
    struct_for_copying()
    {
        struct_for_copying (0, 0, "");
    }
    
    int int_value;
    double double_value;
    std::string string_value;
};

int main (int argc, char **argv)
{
    struct_for_copying input_struct (150 * argc, 10.0 * argc, argv[0]);
    struct_for_copying output_struct;
    int some_int = 44;
    double some_double = 34.5;
    double other_double;
    size_t vector_size;
    std::vector<struct_for_copying> my_vector;
    
    printf ("Here is some code to stop at originally.  Got: %d, %p.\n", argc, argv);
    output_struct = input_struct;
    other_double = (some_double * some_int)/((double) argc);
    other_double = other_double > 0 ? some_double/other_double : some_double > 0 ? other_double/some_double : 10.0;
    my_vector.push_back (input_struct);
    vector_size = my_vector.size();
    
	return vector_size == 0 ? 0 : 1;
}
