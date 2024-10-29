#include <algorithm>
#include <cstdio>
#include <string>

inline int inline_ref_1 (int &value) __attribute__((always_inline));
inline int inline_ref_2 (int &value) __attribute__((always_inline));

int caller_ref_1 (int &value);
int caller_ref_2 (int &value);

int called_by_inline_ref (int &value);

inline void inline_trivial_1 () __attribute__((always_inline));
inline void inline_trivial_2 () __attribute__((always_inline));

void caller_trivial_1 ();
void caller_trivial_2 ();

void called_by_inline_trivial ();

static int inline_value;

int 
function_to_call ()
{
    return inline_value;
}

int
caller_ref_1 (int &value)
{
    int increment = caller_ref_2(value); // In caller_ref_1.
    value += increment; // At increment in caller_ref_1.
    return value;
}

int
caller_ref_2 (int &value)
{
    int increment = inline_ref_1 (value); // In caller_ref_2.
    value += increment;  // At increment in caller_ref_2.
    return value;
}

int
called_by_inline_ref (int &value)
{
    value += 1; // In called_by_inline_ref.
    return value;
}

int
inline_ref_1 (int &value)
{
    int increment = inline_ref_2(value); // In inline_ref_1.
    value += increment; // At increment in inline_ref_1.
    return value;
}

int
inline_ref_2 (int &value)
{
    int increment = called_by_inline_ref (value);  // In inline_ref_2.
    value += 1; // At increment in inline_ref_2.
    return value; 
}

void
caller_trivial_1 ()
{
    caller_trivial_2(); // In caller_trivial_1.
    inline_value += 1; 
}

void
caller_trivial_2 ()
{
    asm volatile ("nop"); inline_trivial_1 (); // In caller_trivial_2.
    inline_value += 1;  // At increment in caller_trivial_2.
}

void
called_by_inline_trivial ()
{
    inline_value += 1; // In called_by_inline_trivial.
}

void
inline_trivial_1 ()
{
    asm volatile ("nop"); inline_trivial_2(); // In inline_trivial_1.
    inline_value += 1;  // At increment in inline_trivial_1.
}

void
inline_trivial_2 ()
{
    inline_value += 1; // In inline_trivial_2.
    called_by_inline_trivial (); // At caller_by_inline_trivial in inline_trivial_2.
}

template<typename T> T
max_value(const T& lhs, const T& rhs)
{
    return std::max(lhs, rhs); // In max_value template
}

template<> std::string
max_value(const std::string& lhs, const std::string& rhs)
{
    return (lhs.size() > rhs.size()) ? lhs : rhs; // In max_value specialized
}

int
main (int argc, char **argv)
{
    
    inline_value = 0;    // Stop here and step over to set up stepping over.

    inline_trivial_1 ();    // At inline_trivial_1 called from main.

    caller_trivial_1();     // At first call of caller_trivial_1 in main.
    
    caller_trivial_1();     // At second call of caller_trivial_1 in main.
    
    caller_ref_1 (argc); // At first call of caller_ref_1 in main.
    
    caller_ref_1 (argc); // At second call of caller_ref_1 in main. 
    
    function_to_call (); // Make sure debug info for this function gets generated.
    
    max_value(123, 456);                                // Call max_value template
    max_value(std::string("abc"), std::string("0022")); // Call max_value specialized

    return 0;            // About to return from main.
}
