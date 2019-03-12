To run the tests, use `py.test <http://pytest.org/latest/>`_::

    py.test tests

The tests are all located in the tests/ directory. To add a new unit
test all you have to do is create the file in the tests/ directory with a
filename in this format::

    test_*.py

New test case classes may wish to inherit from ``PexpectTestCase.PexpectTestCase``
in the tests directory, which sets up some convenient functionality.
