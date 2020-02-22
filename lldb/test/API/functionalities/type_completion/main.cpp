#include <string.h>
#include <vector>
#include <iostream>

class CustomString
{
public:
  CustomString (const char* buffer) :
    m_buffer(nullptr)
  {
    if (buffer)
    {
      auto l = strlen(buffer);
      m_buffer = new char[1 + l];
      strcpy(m_buffer, buffer);
    }
  }
  
  ~CustomString ()
  {
    delete[] m_buffer;
  }
  
  const char*
  GetBuffer ()
  {
    return m_buffer;
  }
  
private:
  char *m_buffer;
};

class NameAndAddress
	{
	public:
		CustomString& GetName() { return *m_name; }
		CustomString& GetAddress() { return *m_address; }
		NameAndAddress(const char* N, const char* A) : m_name(new CustomString(N)), m_address(new CustomString(A))
		{
		}
		~NameAndAddress()
		{
		}
		
	private:
		CustomString* m_name;
		CustomString* m_address;
};

typedef std::vector<NameAndAddress> People;

int main (int argc, const char * argv[])
{
	People p;
	p.push_back(NameAndAddress("Enrico","123 Main Street"));
	p.push_back(NameAndAddress("Foo","10710 Johnson Avenue")); // Set break point at this line.
	p.push_back(NameAndAddress("Arpia","6956 Florey Street"));
	p.push_back(NameAndAddress("Apple","1 Infinite Loop")); // Set break point at this line.
	p.push_back(NameAndAddress("Richard","9500 Gilman Drive"));
	p.push_back(NameAndAddress("Bar","3213 Windsor Rd"));

	for (int j = 0; j<p.size(); j++)
	{
		NameAndAddress guy = p[j];
		std::cout << "Person " << j << " is named " << guy.GetName().GetBuffer() << " and lives at " << guy.GetAddress().GetBuffer() << std::endl; // Set break point at this line.
	}

	return 0;
	
}

