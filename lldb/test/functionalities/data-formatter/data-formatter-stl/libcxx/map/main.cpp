#include <string>
#ifdef _LIBCPP_INLINE_VISIBILITY
#undef _LIBCPP_INLINE_VISIBILITY
#endif
#define _LIBCPP_INLINE_VISIBILITY
#include <map>

#define intint_map std::map<int, int> 
#define strint_map std::map<std::string, int> 
#define intstr_map std::map<int, std::string> 
#define strstr_map std::map<std::string, std::string> 


int main()
{
    intint_map ii;
    
    ii[0] = 0; // Set break point at this line.
    ii[1] = 1;
    ii[2] = 0;
    ii[3] = 1;
    ii[4] = 0;
    ii[5] = 1;
    ii[6] = 0;
    ii[7] = 1;
    ii[85] = 1234567;
    
    ii.clear();
    
    strint_map si;
    
    si["zero"] = 0;
    si["one"] = 1;
    si["two"] = 2;
    si["three"] = 3;
    si["four"] = 4;

    si.clear();
    
    intstr_map is;
    
    is[85] = "goofy";
    is[1] = "is";
    is[2] = "smart";
    is[3] = "!!!";
    
    is.clear();
    
    strstr_map ss;
    
    ss["ciao"] = "hello";
    ss["casa"] = "house";
    ss["gatto"] = "cat";
    ss["a Mac.."] = "..is always a Mac!";
    
    ss.clear();
    
    return 0;
}