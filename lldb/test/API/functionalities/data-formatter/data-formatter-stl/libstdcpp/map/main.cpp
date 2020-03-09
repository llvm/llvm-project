#include <map>
#include <string>

#define intint_map std::map<int, int> 
#define strint_map std::map<std::string, int> 
#define intstr_map std::map<int, std::string> 
#define strstr_map std::map<std::string, std::string> 


int main()
{
    intint_map ii;
    
    ii[0] = 0; // Set break point at this line.
    ii[1] = 1;
    ii[2] = 0;// Set break point at this line.
    ii[3] = 1;
    ii[4] = 0;// Set break point at this line.
    ii[5] = 1;
    ii[6] = 0;
    ii[7] = 1;
    ii[85] = 1234567;
    
    ii.clear();// Set break point at this line.
    
    strint_map si;
    
    si["zero"] = 0;// Set break point at this line.
    si["one"] = 1;// Set break point at this line.
    si["two"] = 2;
    si["three"] = 3;
    si["four"] = 4;

    si.clear();// Set break point at this line.
    
    intstr_map is;
    
    is[85] = "goofy";// Set break point at this line.
    is[1] = "is";
    is[2] = "smart";
    is[3] = "!!!";
    
    is.clear();// Set break point at this line.
    
    strstr_map ss;
    
    ss["ciao"] = "hello";// Set break point at this line.
    ss["casa"] = "house";
    ss["gatto"] = "cat";
    ss["a Mac.."] = "..is always a Mac!";
    
    ss.clear();// Set break point at this line.
    
    return 0;// Set break point at this line.
}
