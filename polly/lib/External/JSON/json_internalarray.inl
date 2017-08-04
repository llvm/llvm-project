// included by json_value.cpp
// everything is within Json namespace

// //////////////////////////////////////////////////////////////////
// //////////////////////////////////////////////////////////////////
// //////////////////////////////////////////////////////////////////
// class ValueInternalArray
// //////////////////////////////////////////////////////////////////
// //////////////////////////////////////////////////////////////////
// //////////////////////////////////////////////////////////////////

ValueArrayAllocator::~ValueArrayAllocator()
{
}

// //////////////////////////////////////////////////////////////////
// class DefaultValueArrayAllocator
// //////////////////////////////////////////////////////////////////
#ifdef JSON_USE_SIMPLE_INTERNAL_ALLOCATOR
class DefaultValueArrayAllocator : public ValueArrayAllocator
{
public: // overridden from ValueArrayAllocator
   virtual ~DefaultValueArrayAllocator()
   {
   }

   virtual ValueInternalArray *newArray()
   {
      return new ValueInternalArray();
   }

   virtual ValueInternalArray *newArrayCopy( const ValueInternalArray &other )
   {
      return new ValueInternalArray( other );
   }

   virtual void destructArray( ValueInternalArray *array )
   {
      delete array;
   }

   virtual void reallocateArrayPageIndex( Value **&indexes, 
                                          ValueInternalArray::PageIndex &indexCount,
                                          ValueInternalArray::PageIndex minNewIndexCount )
   {
      ValueInternalArray::PageIndex newIndexCount = (indexCount*3)/2 + 1;
      if ( minNewIndexCount > newIndexCount )
         newIndexCount = minNewIndexCount;
      void *newIndexes = realloc( indexes, sizeof(Value*) * newIndexCount );
      if ( !newIndexes )
         throw std::bad_alloc();
      indexCount = newIndexCount;
      indexes = static_cast<Value **>( newIndexes );
   }
   virtual void releaseArrayPageIndex( Value **indexes, 
                                       ValueInternalArray::PageIndex indexCount )
   {
      if ( indexes )
         free( indexes );
   }

   virtual Value *allocateArrayPage()
   {
      return static_cast<Value *>( malloc( sizeof(Value) * ValueInternalArray::itemsPerPage ) );
   }

   virtual void releaseArrayPage( Value *value )
   {
      if ( value )
         free( value );
   }
};

#else // #ifdef JSON_USE_SIMPLE_INTERNAL_ALLOCATOR
/// @todo make this thread-safe (lock when accessign batch allocator)
class DefaultValueArrayAllocator : public ValueArrayAllocator
{
public: // overridden from ValueArrayAllocator
   virtual ~DefaultValueArrayAllocator()
   {
   }

   virtual ValueInternalArray *newArray()
   {
      ValueInternalArray *array = arraysAllocator_.allocate();
      new (array) ValueInternalArray(); // placement new
      return array;
   }

   virtual ValueInternalArray *newArrayCopy( const ValueInternalArray &other )
   {
      ValueInternalArray *array = arraysAllocator_.allocate();
      new (array) ValueInternalArray( other ); // placement new
      return array;
   }

   virtual void destructArray( ValueInternalArray *array )
   {
      if ( array )
      {
         array->~ValueInternalArray();
         arraysAllocator_.release( array );
      }
   }

   virtual void reallocateArrayPageIndex( Value **&indexes, 
                                          ValueInternalArray::PageIndex &indexCount,
                                          ValueInternalArray::PageIndex minNewIndexCount )
   {
      ValueInternalArray::PageIndex newIndexCount = (indexCount*3)/2 + 1;
      if ( minNewIndexCount > newIndexCount )
         newIndexCount = minNewIndexCount;
      void *newIndexes = realloc( indexes, sizeof(Value*) * newIndexCount );
      if ( !newIndexes )
         throw std::bad_alloc();
      indexCount = newIndexCount;
      indexes = static_cast<Value **>( newIndexes );
   }
   virtual void releaseArrayPageIndex( Value **indexes, 
                                       ValueInternalArray::PageIndex indexCount )
   {
      if ( indexes )
         free( indexes );
   }

   virtual Value *allocateArrayPage()
   {
      return static_cast<Value *>( pagesAllocator_.allocate() );
   }

   virtual void releaseArrayPage( Value *value )
   {
      if ( value )
         pagesAllocator_.release( value );
   }
private:
   BatchAllocator<ValueInternalArray,1> arraysAllocator_;
   BatchAllocator<Value,ValueInternalArray::itemsPerPage> pagesAllocator_;
};
#endif // #ifdef JSON_USE_SIMPLE_INTERNAL_ALLOCATOR

static ValueArrayAllocator *&arrayAllocator()
{
   static DefaultValueArrayAllocator defaultAllocator;
   static ValueArrayAllocator *arrayAllocator = &defaultAllocator;
   return arrayAllocator;
}

static struct DummyArrayAllocatorInitializer {
   DummyArrayAllocatorInitializer() 
   {
      arrayAllocator();      // ensure arrayAllocator() statics are initialized before main().
   }
} dummyArrayAllocatorInitializer;

// //////////////////////////////////////////////////////////////////
// class ValueInternalArray
// //////////////////////////////////////////////////////////////////
bool 
ValueInternalArray::equals( const IteratorState &x, 
                            const IteratorState &other )
{
   return x.array_ == other.array_  
          &&  x.currentItemIndex_ == other.currentItemIndex_  
          &&  x.currentPageIndex_ == other.currentPageIndex_;
}


void 
ValueInternalArray::increment( IteratorState &it )
{
   JSON_ASSERT_MESSAGE( it.array_  &&
      (it.currentPageIndex_ - it.array_->pages_)*itemsPerPage + it.currentItemIndex_
      != it.array_->size_,
      "ValueInternalArray::increment(): moving iterator beyond end" );
   ++(it.currentItemIndex_);
   if ( it.currentItemIndex_ == itemsPerPage )
   {
      it.currentItemIndex_ = 0;
      ++(it.currentPageIndex_);
   }
}


void 
ValueInternalArray::decrement( IteratorState &it )
{
   JSON_ASSERT_MESSAGE( it.array_  &&  it.currentPageIndex_ == it.array_->pages_ 
                        &&  it.currentItemIndex_ == 0,
      "ValueInternalArray::decrement(): moving iterator beyond end" );
   if ( it.currentItemIndex_ == 0 )
   {
      it.currentItemIndex_ = itemsPerPage-1;
      --(it.currentPageIndex_);
   }
   else
   {
      --(it.currentItemIndex_);
   }
}


Value &
ValueInternalArray::unsafeDereference( const IteratorState &it )
{
   return (*(it.currentPageIndex_))[it.currentItemIndex_];
}


Value &
ValueInternalArray::dereference( const IteratorState &it )
{
   JSON_ASSERT_MESSAGE( it.array_  &&
      (it.currentPageIndex_ - it.array_->pages_)*itemsPerPage + it.currentItemIndex_
      < it.array_->size_,
      "ValueInternalArray::dereference(): dereferencing invalid iterator" );
   return unsafeDereference( it );
}

void 
ValueInternalArray::makeBeginIterator( IteratorState &it ) const
{
   it.array_ = const_cast<ValueInternalArray *>( this );
   it.currentItemIndex_ = 0;
   it.currentPageIndex_ = pages_;
}


void 
ValueInternalArray::makeIterator( IteratorState &it, ArrayIndex index ) const
{
   it.array_ = const_cast<ValueInternalArray *>( this );
   it.currentItemIndex_ = index % itemsPerPage;
   it.currentPageIndex_ = pages_ + index / itemsPerPage;
}


void 
ValueInternalArray::makeEndIterator( IteratorState &it ) const
{
   makeIterator( it, size_ );
}


ValueInternalArray::ValueInternalArray()
   : pages_( 0 )
   , size_( 0 )
   , pageCount_( 0 )
{
}


ValueInternalArray::ValueInternalArray( const ValueInternalArray &other )
   : pages_( 0 )
   , pageCount_( 0 )
   , size_( other.size_ )
{
   PageIndex minNewPages = other.size_ / itemsPerPage;
   arrayAllocator()->reallocateArrayPageIndex( pages_, pageCount_, minNewPages );
   JSON_ASSERT_MESSAGE( pageCount_ >= minNewPages, 
                        "ValueInternalArray::reserve(): bad reallocation" );
   IteratorState itOther;
   other.makeBeginIterator( itOther );
   Value *value;
   for ( ArrayIndex index = 0; index < size_; ++index, increment(itOther) )
   {
      if ( index % itemsPerPage == 0 )
      {
         PageIndex pageIndex = index / itemsPerPage;
         value = arrayAllocator()->allocateArrayPage();
         pages_[pageIndex] = value;
      }
      new (value) Value( dereference( itOther ) );
   }
}


ValueInternalArray &
ValueInternalArray::operator =( const ValueInternalArray &other )
{
   ValueInternalArray temp( other );
   swap( temp );
   return *this;
}


ValueInternalArray::~ValueInternalArray()
{
   // destroy all constructed items
   IteratorState it;
   IteratorState itEnd;
   makeBeginIterator( it);
   makeEndIterator( itEnd );
   for ( ; !equals(it,itEnd); increment(it) )
   {
      Value *value = &dereference(it);
      value->~Value();
   }
   // release all pages
   PageIndex lastPageIndex = size_ / itemsPerPage;
   for ( PageIndex pageIndex = 0; pageIndex < lastPageIndex; ++pageIndex )
      arrayAllocator()->releaseArrayPage( pages_[pageIndex] );
   // release pages index
   arrayAllocator()->releaseArrayPageIndex( pages_, pageCount_ );
}


void 
ValueInternalArray::swap( ValueInternalArray &other )
{
   Value **tempPages = pages_;
   pages_ = other.pages_;
   other.pages_ = tempPages;
   ArrayIndex tempSize = size_;
   size_ = other.size_;
   other.size_ = tempSize;
   PageIndex tempPageCount = pageCount_;
   pageCount_ = other.pageCount_;
   other.pageCount_ = tempPageCount;
}

void 
ValueInternalArray::clear()
{
   ValueInternalArray dummy;
   swap( dummy );
}


void 
ValueInternalArray::resize( ArrayIndex newSize )
{
   if ( newSize == 0 )
      clear();
   else if ( newSize < size_ )
   {
      IteratorState it;
      IteratorState itEnd;
      makeIterator( it, newSize );
      makeIterator( itEnd, size_ );
      for ( ; !equals(it,itEnd); increment(it) )
      {
         Value *value = &dereference(it);
         value->~Value();
      }
      PageIndex pageIndex = (newSize + itemsPerPage - 1) / itemsPerPage;
      PageIndex lastPageIndex = size_ / itemsPerPage;
      for ( ; pageIndex < lastPageIndex; ++pageIndex )
         arrayAllocator()->releaseArrayPage( pages_[pageIndex] );
      size_ = newSize;
   }
   else if ( newSize > size_ )
      resolveReference( newSize );
}


void 
ValueInternalArray::makeIndexValid( ArrayIndex index )
{
   // Need to enlarge page index ?
   if ( index >= pageCount_ * itemsPerPage )
   {
      PageIndex minNewPages = (index + 1) / itemsPerPage;
      arrayAllocator()->reallocateArrayPageIndex( pages_, pageCount_, minNewPages );
      JSON_ASSERT_MESSAGE( pageCount_ >= minNewPages, "ValueInternalArray::reserve(): bad reallocation" );
   }

   // Need to allocate new pages ?
   ArrayIndex nextPageIndex = 
      (size_ % itemsPerPage) != 0 ? size_ - (size_%itemsPerPage) + itemsPerPage
                                  : size_;
   if ( nextPageIndex <= index )
   {
      PageIndex pageIndex = nextPageIndex / itemsPerPage;
      PageIndex pageToAllocate = (index - nextPageIndex) / itemsPerPage + 1;
      for ( ; pageToAllocate-- > 0; ++pageIndex )
         pages_[pageIndex] = arrayAllocator()->allocateArrayPage();
   }

   // Initialize all new entries
   IteratorState it;
   IteratorState itEnd;
   makeIterator( it, size_ );
   size_ = index + 1;
   makeIterator( itEnd, size_ );
   for ( ; !equals(it,itEnd); increment(it) )
   {
      Value *value = &dereference(it);
      new (value) Value(); // Construct a default value using placement new
   }
}

Value &
ValueInternalArray::resolveReference( ArrayIndex index )
{
   if ( index >= size_ )
      makeIndexValid( index );
   return pages_[index/itemsPerPage][index%itemsPerPage];
}

Value *
ValueInternalArray::find( ArrayIndex index ) const
{
   if ( index >= size_ )
      return 0;
   return &(pages_[index/itemsPerPage][index%itemsPerPage]);
}

ValueInternalArray::ArrayIndex 
ValueInternalArray::size() const
{
   return size_;
}

int 
ValueInternalArray::distance( const IteratorState &x, const IteratorState &y )
{
   return indexOf(y) - indexOf(x);
}


ValueInternalArray::ArrayIndex 
ValueInternalArray::indexOf( const IteratorState &iterator )
{
   if ( !iterator.array_ )
      return ArrayIndex(-1);
   return ArrayIndex(
      (iterator.currentPageIndex_ - iterator.array_->pages_) * itemsPerPage 
      + iterator.currentItemIndex_ );
}


int 
ValueInternalArray::compare( const ValueInternalArray &other ) const
{
   int sizeDiff( size_ - other.size_ );
   if ( sizeDiff != 0 )
      return sizeDiff;
   
   for ( ArrayIndex index =0; index < size_; ++index )
   {
      int diff = pages_[index/itemsPerPage][index%itemsPerPage].compare( 
         other.pages_[index/itemsPerPage][index%itemsPerPage] );
      if ( diff != 0 )
         return diff;
   }
   return 0;
}
