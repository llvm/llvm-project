//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <string>
#include <vector>
#include <iostream>

class NameAndAddress
	{
	public:
		std::string& GetName() { return *m_name; }
		std::string& GetAddress() { return *m_address; }
		NameAndAddress(const char* N, const char* A) : m_name(new std::string(N)), m_address(new std::string(A))
		{
		}
		~NameAndAddress()
		{
		}
		
	private:
		std::string* m_name;
		std::string* m_address;
};

typedef std::vector<NameAndAddress> People;

int main (int argc, const char * argv[])
{
	People p;
	p.push_back(NameAndAddress("Enrico","123 Main Street"));
	p.push_back(NameAndAddress("Foo","10710 Johnson Avenue")); // Set break point at this line.
	p.push_back(NameAndAddress("Arpia","6956 Florey Street"));
	p.push_back(NameAndAddress("Apple","1 Infinite Loop"));
	p.push_back(NameAndAddress("Richard","9500 Gilman Drive"));
	p.push_back(NameAndAddress("Bar","3213 Windsor Rd"));

	for (int j = 0; j<p.size(); j++)
	{
		NameAndAddress guy = p[j];
		std::cout << "Person " << j << " is named " << guy.GetName() << " and lives at " << guy.GetAddress() << std::endl;
	}

	return 0;
	
}

