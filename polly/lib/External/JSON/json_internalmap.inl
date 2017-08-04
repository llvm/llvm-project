// included by json_value.cpp
// everything is within Json namespace

// //////////////////////////////////////////////////////////////////
// //////////////////////////////////////////////////////////////////
// //////////////////////////////////////////////////////////////////
// class ValueInternalMap
// //////////////////////////////////////////////////////////////////
// //////////////////////////////////////////////////////////////////
// //////////////////////////////////////////////////////////////////

/** \internal MUST be safely initialized using memset( this, 0, sizeof(ValueInternalLink) );
   * This optimization is used by the fast allocator.
   */
ValueInternalLink::ValueInternalLink()
   : previous_( 0 )
   , next_( 0 )
{
}

ValueInternalLink::~ValueInternalLink()
{ 
   for ( int index =0; index < itemPerLink; ++index )
   {
      if ( !items_[index].isItemAvailable() )
      {
         if ( !items_[index].isMemberNameStatic() )
            free( keys_[index] );
      }
      else
         break;
   }
}



ValueMapAllocator::~ValueMapAllocator()
{
}

#ifdef JSON_USE_SIMPLE_INTERNAL_ALLOCATOR
class DefaultValueMapAllocator : public ValueMapAllocator
{
public: // overridden from ValueMapAllocator
   virtual ValueInternalMap *newMap()
   {
      return new ValueInternalMap();
   }

   virtual ValueInternalMap *newMapCopy( const ValueInternalMap &other )
   {
      return new ValueInternalMap( other );
   }

   virtual void destructMap( ValueInternalMap *map )
   {
      delete map;
   }

   virtual ValueInternalLink *allocateMapBuckets( unsigned int size )
   {
      return new ValueInternalLink[size];
   }

   virtual void releaseMapBuckets( ValueInternalLink *links )
   {
      delete [] links;
   }

   virtual ValueInternalLink *allocateMapLink()
   {
      return new ValueInternalLink();
   }

   virtual void releaseMapLink( ValueInternalLink *link )
   {
      delete link;
   }
};
#else
/// @todo make this thread-safe (lock when accessign batch allocator)
class DefaultValueMapAllocator : public ValueMapAllocator
{
public: // overridden from ValueMapAllocator
   virtual ValueInternalMap *newMap()
   {
      ValueInternalMap *map = mapsAllocator_.allocate();
      new (map) ValueInternalMap(); // placement new
      return map;
   }

   virtual ValueInternalMap *newMapCopy( const ValueInternalMap &other )
   {
      ValueInternalMap *map = mapsAllocator_.allocate();
      new (map) ValueInternalMap( other ); // placement new
      return map;
   }

   virtual void destructMap( ValueInternalMap *map )
   {
      if ( map )
      {
         map->~ValueInternalMap();
         mapsAllocator_.release( map );
      }
   }

   virtual ValueInternalLink *allocateMapBuckets( unsigned int size )
   {
      return new ValueInternalLink[size];
   }

   virtual void releaseMapBuckets( ValueInternalLink *links )
   {
      delete [] links;
   }

   virtual ValueInternalLink *allocateMapLink()
   {
      ValueInternalLink *link = linksAllocator_.allocate();
      memset( link, 0, sizeof(ValueInternalLink) );
      return link;
   }

   virtual void releaseMapLink( ValueInternalLink *link )
   {
      link->~ValueInternalLink();
      linksAllocator_.release( link );
   }
private:
   BatchAllocator<ValueInternalMap,1> mapsAllocator_;
   BatchAllocator<ValueInternalLink,1> linksAllocator_;
};
#endif

static ValueMapAllocator *&mapAllocator()
{
   static DefaultValueMapAllocator defaultAllocator;
   static ValueMapAllocator *mapAllocator = &defaultAllocator;
   return mapAllocator;
}

static struct DummyMapAllocatorInitializer {
   DummyMapAllocatorInitializer() 
   {
      mapAllocator();      // ensure mapAllocator() statics are initialized before main().
   }
} dummyMapAllocatorInitializer;



// h(K) = value * K >> w ; with w = 32 & K prime w.r.t. 2^32.

/*
use linked list hash map. 
buckets array is a container.
linked list element contains 6 key/values. (memory = (16+4) * 6 + 4 = 124)
value have extra state: valid, available, deleted
*/


ValueInternalMap::ValueInternalMap()
   : buckets_( 0 )
   , tailLink_( 0 )
   , bucketsSize_( 0 )
   , itemCount_( 0 )
{
}


ValueInternalMap::ValueInternalMap( const ValueInternalMap &other )
   : buckets_( 0 )
   , tailLink_( 0 )
   , bucketsSize_( 0 )
   , itemCount_( 0 )
{
   reserve( other.itemCount_ );
   IteratorState it;
   IteratorState itEnd;
   other.makeBeginIterator( it );
   other.makeEndIterator( itEnd );
   for ( ; !equals(it,itEnd); increment(it) )
   {
      bool isStatic;
      const char *memberName = key( it, isStatic );
      const Value &aValue = value( it );
      resolveReference(memberName, isStatic) = aValue;
   }
}


ValueInternalMap &
ValueInternalMap::operator =( const ValueInternalMap &other )
{
   ValueInternalMap dummy( other );
   swap( dummy );
   return *this;
}


ValueInternalMap::~ValueInternalMap()
{
   if ( buckets_ )
   {
      for ( BucketIndex bucketIndex =0; bucketIndex < bucketsSize_; ++bucketIndex )
      {
         ValueInternalLink *link = buckets_[bucketIndex].next_;
         while ( link )
         {
            ValueInternalLink *linkToRelease = link;
            link = link->next_;
            mapAllocator()->releaseMapLink( linkToRelease );
         }
      }
      mapAllocator()->releaseMapBuckets( buckets_ );
   }
}


void 
ValueInternalMap::swap( ValueInternalMap &other )
{
   ValueInternalLink *tempBuckets = buckets_;
   buckets_ = other.buckets_;
   other.buckets_ = tempBuckets;
   ValueInternalLink *tempTailLink = tailLink_;
   tailLink_ = other.tailLink_;
   other.tailLink_ = tempTailLink;
   BucketIndex tempBucketsSize = bucketsSize_;
   bucketsSize_ = other.bucketsSize_;
   other.bucketsSize_ = tempBucketsSize;
   BucketIndex tempItemCount = itemCount_;
   itemCount_ = other.itemCount_;
   other.itemCount_ = tempItemCount;
}


void 
ValueInternalMap::clear()
{
   ValueInternalMap dummy;
   swap( dummy );
}


ValueInternalMap::BucketIndex 
ValueInternalMap::size() const
{
   return itemCount_;
}

bool 
ValueInternalMap::reserveDelta( BucketIndex growth )
{
   return reserve( itemCount_ + growth );
}

bool 
ValueInternalMap::reserve( BucketIndex newItemCount )
{
   if ( !buckets_  &&  newItemCount > 0 )
   {
      buckets_ = mapAllocator()->allocateMapBuckets( 1 );
      bucketsSize_ = 1;
      tailLink_ = &buckets_[0];
   }
//   BucketIndex idealBucketCount = (newItemCount + ValueInternalLink::itemPerLink) / ValueInternalLink::itemPerLink;
   return true;
}


const Value *
ValueInternalMap::find( const char *key ) const
{
   if ( !bucketsSize_ )
      return 0;
   HashKey hashedKey = hash( key );
   BucketIndex bucketIndex = hashedKey % bucketsSize_;
   for ( const ValueInternalLink *current = &buckets_[bucketIndex]; 
         current != 0; 
         current = current->next_ )
   {
      for ( BucketIndex index=0; index < ValueInternalLink::itemPerLink; ++index )
      {
         if ( current->items_[index].isItemAvailable() )
            return 0;
         if ( strcmp( key, current->keys_[index] ) == 0 )
            return &current->items_[index];
      }
   }
   return 0;
}


Value *
ValueInternalMap::find( const char *key )
{
   const ValueInternalMap *constThis = this;
   return const_cast<Value *>( constThis->find( key ) );
}


Value &
ValueInternalMap::resolveReference( const char *key,
                                    bool isStatic )
{
   HashKey hashedKey = hash( key );
   if ( bucketsSize_ )
   {
      BucketIndex bucketIndex = hashedKey % bucketsSize_;
      ValueInternalLink **previous = 0;
      BucketIndex index;
      for ( ValueInternalLink *current = &buckets_[bucketIndex]; 
            current != 0; 
            previous = &current->next_, current = current->next_ )
      {
         for ( index=0; index < ValueInternalLink::itemPerLink; ++index )
         {
            if ( current->items_[index].isItemAvailable() )
               return setNewItem( key, isStatic, current, index );
            if ( strcmp( key, current->keys_[index] ) == 0 )
               return current->items_[index];
         }
      }
   }

   reserveDelta( 1 );
   return unsafeAdd( key, isStatic, hashedKey );
}


void 
ValueInternalMap::remove( const char *key )
{
   HashKey hashedKey = hash( key );
   if ( !bucketsSize_ )
      return;
   BucketIndex bucketIndex = hashedKey % bucketsSize_;
   for ( ValueInternalLink *link = &buckets_[bucketIndex]; 
         link != 0; 
         link = link->next_ )
   {
      BucketIndex index;
      for ( index =0; index < ValueInternalLink::itemPerLink; ++index )
      {
         if ( link->items_[index].isItemAvailable() )
            return;
         if ( strcmp( key, link->keys_[index] ) == 0 )
         {
            doActualRemove( link, index, bucketIndex );
            return;
         }
      }
   }
}

void 
ValueInternalMap::doActualRemove( ValueInternalLink *link, 
                                  BucketIndex index,
                                  BucketIndex bucketIndex )
{
   // find last item of the bucket and swap it with the 'removed' one.
   // set removed items flags to 'available'.
   // if last page only contains 'available' items, then desallocate it (it's empty)
   ValueInternalLink *&lastLink = getLastLinkInBucket( index );
   BucketIndex lastItemIndex = 1; // a link can never be empty, so start at 1
   for ( ;   
         lastItemIndex < ValueInternalLink::itemPerLink; 
         ++lastItemIndex ) // may be optimized with dicotomic search
   {
      if ( lastLink->items_[lastItemIndex].isItemAvailable() )
         break;
   }
   
   BucketIndex lastUsedIndex = lastItemIndex - 1;
   Value *valueToDelete = &link->items_[index];
   Value *valueToPreserve = &lastLink->items_[lastUsedIndex];
   if ( valueToDelete != valueToPreserve )
      valueToDelete->swap( *valueToPreserve );
   if ( lastUsedIndex == 0 )  // page is now empty
   {  // remove it from bucket linked list and delete it.
      ValueInternalLink *linkPreviousToLast = lastLink->previous_;
      if ( linkPreviousToLast != 0 )   // can not deleted bucket link.
      {
         mapAllocator()->releaseMapLink( lastLink );
         linkPreviousToLast->next_ = 0;
         lastLink = linkPreviousToLast;
      }
   }
   else
   {
      Value dummy;
      valueToPreserve->swap( dummy ); // restore deleted to default Value.
      valueToPreserve->setItemUsed( false );
   }
   --itemCount_;
}


ValueInternalLink *&
ValueInternalMap::getLastLinkInBucket( BucketIndex bucketIndex )
{
   if ( bucketIndex == bucketsSize_ - 1 )
      return tailLink_;
   ValueInternalLink *&previous = buckets_[bucketIndex+1].previous_;
   if ( !previous )
      previous = &buckets_[bucketIndex];
   return previous;
}


Value &
ValueInternalMap::setNewItem( const char *key, 
                              bool isStatic,
                              ValueInternalLink *link, 
                              BucketIndex index )
{
   char *duplicatedKey = valueAllocator()->makeMemberName( key );
   ++itemCount_;
   link->keys_[index] = duplicatedKey;
   link->items_[index].setItemUsed();
   link->items_[index].setMemberNameIsStatic( isStatic );
   return link->items_[index]; // items already default constructed.
}


Value &
ValueInternalMap::unsafeAdd( const char *key, 
                             bool isStatic, 
                             HashKey hashedKey )
{
   JSON_ASSERT_MESSAGE( bucketsSize_ > 0, "ValueInternalMap::unsafeAdd(): internal logic error." );
   BucketIndex bucketIndex = hashedKey % bucketsSize_;
   ValueInternalLink *&previousLink = getLastLinkInBucket( bucketIndex );
   ValueInternalLink *link = previousLink;
   BucketIndex index;
   for ( index =0; index < ValueInternalLink::itemPerLink; ++index )
   {
      if ( link->items_[index].isItemAvailable() )
         break;
   }
   if ( index == ValueInternalLink::itemPerLink ) // need to add a new page
   {
      ValueInternalLink *newLink = mapAllocator()->allocateMapLink();
      index = 0;
      link->next_ = newLink;
      previousLink = newLink;
      link = newLink;
   }
   return setNewItem( key, isStatic, link, index );
}


ValueInternalMap::HashKey 
ValueInternalMap::hash( const char *key ) const
{
   HashKey hash = 0;
   while ( *key )
      hash += *key++ * 37;
   return hash;
}


int 
ValueInternalMap::compare( const ValueInternalMap &other ) const
{
   int sizeDiff( itemCount_ - other.itemCount_ );
   if ( sizeDiff != 0 )
      return sizeDiff;
   // Strict order guaranty is required. Compare all keys FIRST, then compare values.
   IteratorState it;
   IteratorState itEnd;
   makeBeginIterator( it );
   makeEndIterator( itEnd );
   for ( ; !equals(it,itEnd); increment(it) )
   {
      if ( !other.find( key( it ) ) )
         return 1;
   }

   // All keys are equals, let's compare values
   makeBeginIterator( it );
   for ( ; !equals(it,itEnd); increment(it) )
   {
      const Value *otherValue = other.find( key( it ) );
      int valueDiff = value(it).compare( *otherValue );
      if ( valueDiff != 0 )
         return valueDiff;
   }
   return 0;
}


void 
ValueInternalMap::makeBeginIterator( IteratorState &it ) const
{
   it.map_ = const_cast<ValueInternalMap *>( this );
   it.bucketIndex_ = 0;
   it.itemIndex_ = 0;
   it.link_ = buckets_;
}


void 
ValueInternalMap::makeEndIterator( IteratorState &it ) const
{
   it.map_ = const_cast<ValueInternalMap *>( this );
   it.bucketIndex_ = bucketsSize_;
   it.itemIndex_ = 0;
   it.link_ = 0;
}


bool 
ValueInternalMap::equals( const IteratorState &x, const IteratorState &other )
{
   return x.map_ == other.map_  
          &&  x.bucketIndex_ == other.bucketIndex_  
          &&  x.link_ == other.link_
          &&  x.itemIndex_ == other.itemIndex_;
}


void 
ValueInternalMap::incrementBucket( IteratorState &iterator )
{
   ++iterator.bucketIndex_;
   JSON_ASSERT_MESSAGE( iterator.bucketIndex_ <= iterator.map_->bucketsSize_,
      "ValueInternalMap::increment(): attempting to iterate beyond end." );
   if ( iterator.bucketIndex_ == iterator.map_->bucketsSize_ )
      iterator.link_ = 0;
   else
      iterator.link_ = &(iterator.map_->buckets_[iterator.bucketIndex_]);
   iterator.itemIndex_ = 0;
}


void 
ValueInternalMap::increment( IteratorState &iterator )
{
   JSON_ASSERT_MESSAGE( iterator.map_, "Attempting to iterator using invalid iterator." );
   ++iterator.itemIndex_;
   if ( iterator.itemIndex_ == ValueInternalLink::itemPerLink )
   {
      JSON_ASSERT_MESSAGE( iterator.link_ != 0,
         "ValueInternalMap::increment(): attempting to iterate beyond end." );
      iterator.link_ = iterator.link_->next_;
      if ( iterator.link_ == 0 )
         incrementBucket( iterator );
   }
   else if ( iterator.link_->items_[iterator.itemIndex_].isItemAvailable() )
   {
      incrementBucket( iterator );
   }
}


void 
ValueInternalMap::decrement( IteratorState &iterator )
{
   if ( iterator.itemIndex_ == 0 )
   {
      JSON_ASSERT_MESSAGE( iterator.map_, "Attempting to iterate using invalid iterator." );
      if ( iterator.link_ == &iterator.map_->buckets_[iterator.bucketIndex_] )
      {
         JSON_ASSERT_MESSAGE( iterator.bucketIndex_ > 0, "Attempting to iterate beyond beginning." );
         --(iterator.bucketIndex_);
      }
      iterator.link_ = iterator.link_->previous_;
      iterator.itemIndex_ = ValueInternalLink::itemPerLink - 1;
   }
}


const char *
ValueInternalMap::key( const IteratorState &iterator )
{
   JSON_ASSERT_MESSAGE( iterator.link_, "Attempting to iterate using invalid iterator." );
   return iterator.link_->keys_[iterator.itemIndex_];
}

const char *
ValueInternalMap::key( const IteratorState &iterator, bool &isStatic )
{
   JSON_ASSERT_MESSAGE( iterator.link_, "Attempting to iterate using invalid iterator." );
   isStatic = iterator.link_->items_[iterator.itemIndex_].isMemberNameStatic();
   return iterator.link_->keys_[iterator.itemIndex_];
}


Value &
ValueInternalMap::value( const IteratorState &iterator )
{
   JSON_ASSERT_MESSAGE( iterator.link_, "Attempting to iterate using invalid iterator." );
   return iterator.link_->items_[iterator.itemIndex_];
}


int 
ValueInternalMap::distance( const IteratorState &x, const IteratorState &y )
{
   int offset = 0;
   IteratorState it = x;
   while ( !equals( it, y ) )
      increment( it );
   return offset;
}
