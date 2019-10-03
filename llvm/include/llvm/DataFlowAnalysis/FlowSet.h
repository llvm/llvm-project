#ifndef LLVM_DFA_FLOWSET_H
#define LLVM_DFA_FLOWSET_H
#include <iostream>

using namespace std;
namespace llvm{

template <class T>
class FlowSet
{
    class Node;
public:
    FlowSet() noexcept
    {
	m_spRoot = nullptr;
    }

    class Iterator;
    
    Iterator begin()
    {
	return Iterator(m_spRoot);
    }

    Iterator end()
    {
	return Iterator(nullptr);
    }

    void push_back(T data);

    class Iterator
    {
    public:
	Iterator() noexcept:
	    m_pCurrentNode(m_spRoot){   }

	Iterator(const Node* pNode) noexcept:
	    m_pCurrentNode(pNode){  }

	Iterator& operator=(Node* pNode)
	{
	    this->m_pCurrentNode = pNode;
	    return *this;
	}

	bool operator!=(const Iterator& it)
	{
	    return this->m_pCurrentNode != it.m_pCurrentNode;
	}

	bool operator==(const Iterator& it)
	{
	    return this->m_pCurrentNode == it.m_pCurrentNode;
	}

	Iterator& operator++()
	{
	    if(m_pCurrentNode)
		m_pCurrentNode = m_pCurrentNode->pNext;
	    return *this;
	}

	//  post increment (must be non reference)
	Iterator operator++(int)
	{
	    Iterator it = *this;
	    ++(*this);
	    return it;
	}

	int operator*()
	{
	    return m_pCurrentNode->data;
	}
    private:
	const Node* m_pCurrentNode;
    };
private:
    class Node
    {
	T data;
	Node* pNext;
	
	friend class FlowSet;
    };

    Node* getNode(T data)
    {
	Node* pNewNode = new Node;
	pNewNode->data = data;
	pNewNode->pNext = nullptr;

	return pNewNode;
    }

    Node*& getRootNode()
    {
	return m_spRoot;
    }

    static Node* m_spRoot;
};

template <typename T>
typename FlowSet<T>::Node* FlowSet<T>::m_spRoot = nullptr;
/*
template <typename T>
void FlowSet<T>::push_back(T data)
{
    Node* pTemp = getNode(data);
    if(!getRootNode())
    {
	getRootNode() = pTemp; 
    }
    else
    {
	Node* pC = getRootNode();
	while(pC->pNext)
	{
	    pC = pC->pNext;
	}
	pC->pNext = pTemp;
    }
}
*/
}   // end namepspace llvm

#endif	// LLVM_DFA_FLOWSET_H
