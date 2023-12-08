// This file must have the following defined before it is included:
// T defined to the type to test (int, float, etc)
// T_CSTR a C string representation of the type T ("int", "float")
// T_VALUE_1 defined to a valid initializer value for TEST_TYPE (7 for int, 2.0 for float)
// T_VALUE_2, T_VALUE_3, T_VALUE_4 defined to a valid initializer value for TEST_TYPE that is different from TEST_VALUE_1
// T_PRINTF_FORMAT defined if T can be printed with printf
//
// An example for integers is below
#if 0

#define T int
#define T_CSTR "int"
#define T_VALUE_1 11001110
#define T_VALUE_2 22002220
#define T_VALUE_3 33003330
#define T_VALUE_4 44044440
#define T_PRINTF_FORMAT "%i"

#include "basic_type.cpp"

#endif

#ifdef TEST_BLOCK_CAPTURED_VARS
#include <dispatch/dispatch.h>
#endif
#include <cstdint>
#include <cstdio>
#include <cstdlib>

class a_class 
{
public:
    a_class (const T& a, const T& b) :
        m_a (a),
        m_b (b)
    {
    }

    ~a_class ()
    {
    }

    const T&
    get_a() const
    {
        return m_a;
    } 

    void
    set_a (const T& a)
    {
        m_a = a;
    }

    const T&
    get_b() const
    {
        return m_b;
    } 

    void
    set_b (const T& b)
    {
        m_b = b;
    }

protected:
    T m_a;
    T m_b;
};

typedef struct a_struct_tag {
    T a;
    T b;
} a_struct_t;


typedef union a_union_zero_tag {
    T a;
    double a_double;
} a_union_zero_t;

typedef struct a_union_nonzero_tag {
  double a_double;
  a_union_zero_t u;
} a_union_nonzero_t;


int 
main (int argc, char const *argv[])
{
    FILE *out = stdout;

    // By default, output to stdout
    // If a filename is provided as the command line argument,
    // output to that file.
    if (argc == 2 && argv[1] && argv[1][0] != '\0')
    {
        out = fopen (argv[1], "w");
    }

    T a = T_VALUE_1;
    T* a_ptr = &a;
    T& a_ref = a;
    T a_array_bounded[2] = { T_VALUE_1, T_VALUE_2 };    
    T a_array_unbounded[] = { T_VALUE_1, T_VALUE_2 };

    a_class a_class_instance (T_VALUE_1, T_VALUE_2);
    a_class *a_class_ptr = &a_class_instance;
    a_class &a_class_ref = a_class_instance;

    a_struct_t a_struct = { T_VALUE_1, T_VALUE_2 };
    a_struct_t *a_struct_ptr = &a_struct;
    a_struct_t &a_struct_ref = a_struct;

    // Create a union with type T at offset zero
    a_union_zero_t a_union_zero;
    a_union_zero.a = T_VALUE_1;
    a_union_zero_t *a_union_zero_ptr = &a_union_zero;
    a_union_zero_t &a_union_zero_ref = a_union_zero;

    // Create a union with type T at a non-zero offset
    a_union_nonzero_t a_union_nonzero;
    a_union_nonzero.u.a = T_VALUE_1;
    a_union_nonzero_t *a_union_nonzero_ptr = &a_union_nonzero;
    a_union_nonzero_t &a_union_nonzero_ref = a_union_nonzero;

    a_struct_t a_struct_array_bounded[2]  = {{ T_VALUE_1, T_VALUE_2 }, { T_VALUE_3, T_VALUE_4 }};
    a_struct_t a_struct_array_unbounded[] = {{ T_VALUE_1, T_VALUE_2 }, { T_VALUE_3, T_VALUE_4 }};
    a_union_zero_t a_union_zero_array_bounded[2];
    a_union_zero_array_bounded[0].a = T_VALUE_1;
    a_union_zero_array_bounded[1].a = T_VALUE_2;
    a_union_zero_t a_union_zero_array_unbounded[] = {{ T_VALUE_1 }, { T_VALUE_2 }};
    
#ifdef T_PRINTF_FORMAT
    fprintf (out, "%s: a = '" T_PRINTF_FORMAT "'\n", T_CSTR, a);
    fprintf (out, "%s*: %p => *a_ptr = '" T_PRINTF_FORMAT "'\n", T_CSTR, a_ptr, *a_ptr);
    fprintf (out, "%s&: @%p => a_ref = '" T_PRINTF_FORMAT "'\n", T_CSTR, &a_ref, a_ref);

    fprintf (out, "%s[2]: a_array_bounded[0] = '" T_PRINTF_FORMAT "'\n", T_CSTR, a_array_bounded[0]);
    fprintf (out, "%s[2]: a_array_bounded[1] = '" T_PRINTF_FORMAT "'\n", T_CSTR, a_array_bounded[1]);

    fprintf (out, "%s[]: a_array_unbounded[0] = '" T_PRINTF_FORMAT "'\n", T_CSTR, a_array_unbounded[0]);
    fprintf (out, "%s[]: a_array_unbounded[1] = '" T_PRINTF_FORMAT "'\n", T_CSTR, a_array_unbounded[1]);

    fprintf (out, "(a_class) a_class_instance.m_a = '" T_PRINTF_FORMAT "'\n", a_class_instance.get_a());
    fprintf (out, "(a_class) a_class_instance.m_b = '" T_PRINTF_FORMAT "'\n", a_class_instance.get_b());
    fprintf (out, "(a_class*) a_class_ptr = %p, a_class_ptr->m_a = '" T_PRINTF_FORMAT "'\n", a_class_ptr, a_class_ptr->get_a());
    fprintf (out, "(a_class*) a_class_ptr = %p, a_class_ptr->m_b = '" T_PRINTF_FORMAT "'\n", a_class_ptr, a_class_ptr->get_b());
    fprintf (out, "(a_class&) a_class_ref = %p, a_class_ref.m_a = '" T_PRINTF_FORMAT "'\n", &a_class_ref, a_class_ref.get_a());
    fprintf (out, "(a_class&) a_class_ref = %p, a_class_ref.m_b = '" T_PRINTF_FORMAT "'\n", &a_class_ref, a_class_ref.get_b());

    fprintf (out, "(a_struct_t) a_struct.a = '" T_PRINTF_FORMAT "'\n", a_struct.a);
    fprintf (out, "(a_struct_t) a_struct.b = '" T_PRINTF_FORMAT "'\n", a_struct.b);
    fprintf (out, "(a_struct_t*) a_struct_ptr = %p, a_struct_ptr->a = '" T_PRINTF_FORMAT "'\n", a_struct_ptr, a_struct_ptr->a);
    fprintf (out, "(a_struct_t*) a_struct_ptr = %p, a_struct_ptr->b = '" T_PRINTF_FORMAT "'\n", a_struct_ptr, a_struct_ptr->b);
    fprintf (out, "(a_struct_t&) a_struct_ref = %p, a_struct_ref.a = '" T_PRINTF_FORMAT "'\n", &a_struct_ref, a_struct_ref.a);
    fprintf (out, "(a_struct_t&) a_struct_ref = %p, a_struct_ref.b = '" T_PRINTF_FORMAT "'\n", &a_struct_ref, a_struct_ref.b);
    
    fprintf (out, "(a_union_zero_t) a_union_zero.a = '" T_PRINTF_FORMAT "'\n", a_union_zero.a);
    fprintf (out, "(a_union_zero_t*) a_union_zero_ptr = %p, a_union_zero_ptr->a = '" T_PRINTF_FORMAT "'\n", a_union_zero_ptr, a_union_zero_ptr->a);
    fprintf (out, "(a_union_zero_t&) a_union_zero_ref = %p, a_union_zero_ref.a = '" T_PRINTF_FORMAT "'\n", &a_union_zero_ref, a_union_zero_ref.a);

    fprintf (out, "(a_union_nonzero_t) a_union_nonzero.u.a = '" T_PRINTF_FORMAT "'\n", a_union_nonzero.u.a);
    fprintf (out, "(a_union_nonzero_t*) a_union_nonzero_ptr = %p, a_union_nonzero_ptr->u.a = '" T_PRINTF_FORMAT "'\n", a_union_nonzero_ptr, a_union_nonzero_ptr->u.a);
    fprintf (out, "(a_union_nonzero_t&) a_union_nonzero_ref = %p, a_union_nonzero_ref.u.a = '" T_PRINTF_FORMAT "'\n", &a_union_nonzero_ref, a_union_nonzero_ref.u.a);

    fprintf (out, "(a_struct_t[2]) a_struct_array_bounded[0].a = '" T_PRINTF_FORMAT "'\n", a_struct_array_bounded[0].a);
    fprintf (out, "(a_struct_t[2]) a_struct_array_bounded[0].b = '" T_PRINTF_FORMAT "'\n", a_struct_array_bounded[0].b);
    fprintf (out, "(a_struct_t[2]) a_struct_array_bounded[1].a = '" T_PRINTF_FORMAT "'\n", a_struct_array_bounded[1].a);
    fprintf (out, "(a_struct_t[2]) a_struct_array_bounded[1].b = '" T_PRINTF_FORMAT "'\n", a_struct_array_bounded[1].b);

    fprintf (out, "(a_struct_t[]) a_struct_array_unbounded[0].a = '" T_PRINTF_FORMAT "'\n", a_struct_array_unbounded[0].a);
    fprintf (out, "(a_struct_t[]) a_struct_array_unbounded[0].b = '" T_PRINTF_FORMAT "'\n", a_struct_array_unbounded[0].b);
    fprintf (out, "(a_struct_t[]) a_struct_array_unbounded[1].a = '" T_PRINTF_FORMAT "'\n", a_struct_array_unbounded[1].a);
    fprintf (out, "(a_struct_t[]) a_struct_array_unbounded[1].b = '" T_PRINTF_FORMAT "'\n", a_struct_array_unbounded[1].b);

    fprintf (out, "(a_union_zero_t[2]) a_union_zero_array_bounded[0].a = '" T_PRINTF_FORMAT "'\n", a_union_zero_array_bounded[0].a);
    fprintf (out, "(a_union_zero_t[2]) a_union_zero_array_bounded[1].a = '" T_PRINTF_FORMAT "'\n", a_union_zero_array_bounded[1].a);

    fprintf (out, "(a_union_zero_t[]) a_union_zero_array_unbounded[0].a = '" T_PRINTF_FORMAT "'\n", a_union_zero_array_unbounded[0].a);
    fprintf (out, "(a_union_zero_t[]) a_union_zero_array_unbounded[1].a = '" T_PRINTF_FORMAT "'\n", a_union_zero_array_unbounded[1].a);

#endif
    puts("About to exit, break here to check values..."); // Here is the line we will break on to check variables.

#ifdef TEST_BLOCK_CAPTURED_VARS
    void (^myBlock)() = ^() {
        fprintf (out, "%s: a = '" T_PRINTF_FORMAT "'\n", T_CSTR, a);
        fprintf (out, "%s*: %p => *a_ptr = '" T_PRINTF_FORMAT "'\n", T_CSTR, a_ptr, *a_ptr);
        fprintf (out, "%s&: @%p => a_ref = '" T_PRINTF_FORMAT "'\n", T_CSTR, &a_ref, a_ref);

        fprintf (out, "(a_class) a_class_instance.m_a = '" T_PRINTF_FORMAT "'\n", a_class_instance.get_a());
        fprintf (out, "(a_class) a_class_instance.m_b = '" T_PRINTF_FORMAT "'\n", a_class_instance.get_b());
        fprintf (out, "(a_class*) a_class_ptr = %p, a_class_ptr->m_a = '" T_PRINTF_FORMAT "'\n", a_class_ptr, a_class_ptr->get_a());
        fprintf (out, "(a_class*) a_class_ptr = %p, a_class_ptr->m_b = '" T_PRINTF_FORMAT "'\n", a_class_ptr, a_class_ptr->get_b());
        fprintf (out, "(a_class&) a_class_ref = %p, a_class_ref.m_a = '" T_PRINTF_FORMAT "'\n", &a_class_ref, a_class_ref.get_a());
        fprintf (out, "(a_class&) a_class_ref = %p, a_class_ref.m_b = '" T_PRINTF_FORMAT "'\n", &a_class_ref, a_class_ref.get_b());
    
        fprintf (out, "(a_struct_t) a_struct.a = '" T_PRINTF_FORMAT "'\n", a_struct.a);
        fprintf (out, "(a_struct_t) a_struct.b = '" T_PRINTF_FORMAT "'\n", a_struct.b);
        fprintf (out, "(a_struct_t*) a_struct_ptr = %p, a_struct_ptr->a = '" T_PRINTF_FORMAT "'\n", a_struct_ptr, a_struct_ptr->a);
        fprintf (out, "(a_struct_t*) a_struct_ptr = %p, a_struct_ptr->b = '" T_PRINTF_FORMAT "'\n", a_struct_ptr, a_struct_ptr->b);
        fprintf (out, "(a_struct_t&) a_struct_ref = %p, a_struct_ref.a = '" T_PRINTF_FORMAT "'\n", &a_struct_ref, a_struct_ref.a);
        fprintf (out, "(a_struct_t&) a_struct_ref = %p, a_struct_ref.b = '" T_PRINTF_FORMAT "'\n", &a_struct_ref, a_struct_ref.b);
    
        fprintf (out, "(a_union_zero_t) a_union_zero.a = '" T_PRINTF_FORMAT "'\n", a_union_zero.a);
        fprintf (out, "(a_union_zero_t*) a_union_zero_ptr = %p, a_union_zero_ptr->a = '" T_PRINTF_FORMAT "'\n", a_union_zero_ptr, a_union_zero_ptr->a);
        fprintf (out, "(a_union_zero_t&) a_union_zero_ref = %p, a_union_zero_ref.a = '" T_PRINTF_FORMAT "'\n", &a_union_zero_ref, a_union_zero_ref.a);

        fprintf (out, "(a_union_nonzero_t) a_union_nonzero.u.a = '" T_PRINTF_FORMAT "'\n", a_union_nonzero.u.a);
        fprintf (out, "(a_union_nonzero_t*) a_union_nonzero_ptr = %p, a_union_nonzero_ptr->u.a = '" T_PRINTF_FORMAT "'\n", a_union_nonzero_ptr, a_union_nonzero_ptr->u.a);
        fprintf (out, "(a_union_nonzero_t&) a_union_nonzero_ref = %p, a_union_nonzero_ref.u.a = '" T_PRINTF_FORMAT "'\n", &a_union_nonzero_ref, a_union_nonzero_ref.u.a);

        fprintf (out, "That's All Folks!\n"); // Break here to test block captured variables.
    };

    myBlock();
#endif

    if (out != stdout)
        fclose (out);

    return 0;
}
