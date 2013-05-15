import json
from isl import *

class Scop:
  def __init__(self, filename):
    f = open(filename, 'r')
    self.json = json.load(f)
    return

  def __str__(self):
    return json.dumps(self.json, indent=2)

  def __repr__(self):
    return str(self)

  @property
  def statements(self):
    return self.json['statements']

class Transforms:
  """
  Create a map that interchanges two dimensions 'A' and 'B'

  numberDimensions: The overall number of dimensions
  dimensionA: The dimension of dimension 'A'
  dimensionB: The dimension of dimension 'B'

  getInterchange(2, 0, 1):
  {[d0, d1] -> [d1, d0]}
  """
  @staticmethod
  def getInterchange(numberDimensions, dimensionA, dimensionB):

    dims = ['d' + str(i) for i in range(numberDimensions)]
    dimString = ",".join(dims)

    changedDims = dims
    first = dims[dimensionA]
    second = dims[dimensionB]
    changedDims[dimensionA] = second
    changedDims[dimensionB] = first
    changedDimString = ",".join(changedDims)

    return Map("{[%s] -> [%s]}" % (dimString, changedDimString))

  """
  Create a map that strip mines one dimension

  numberDimensions: The overall number of dimensions
  stripMineDim: The dimension to strip mine
  factor: The strip mining factor

  getStripMine(2, 1, 64):
  {[d0, d1] -> [d0, o, d1] : o % 64 = 0 and o <= d1 <= d1 + 63}
  """
  @staticmethod
  def getStripMine(numberDimensions, stripMineDim, factor):

    dims = ['d' + str(i) for i in range(numberDimensions)]
    dimString = ",".join(dims)

    changedDims = dims
    smd = dims[stripMineDim]
    changedDims[stripMineDim] = "o,%s" % smd
    changedDimString = ",".join(changedDims)
    string = "{[%s] -> [%s]: o %% %i = 0 and o <= %s <= o + %i}" % \
          (dimString, changedDimString, factor, smd, factor - 1)
    return Map(string)
